import sys
import dataclasses
import typing as tp
from logging import getLogger

import torch

sys.path.append('.')
sys.path.append('../')
from core.transcoder.dataloaders.preprocessing.dictionary import UNK_WORD
from core.transcoder.utils import get_parser
from core.transcoder.pretrain import load_embeddings
from core.transcoder.transformer import DECODER_ONLY_PARAMS, TransformerModel


logger = getLogger(__name__)


class AttrDict(dict):
    def __init__(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattr__(self, name: str) -> tp.Any:  # deactivates mypy checks
        raise RuntimeError


def add_missing_parameters(parameters: tp.Any, log: bool = True) -> None:
    """
        Adds missing default arguments into the parameters object
        This only applies to AttrDict instances which mock the parsed args
        when reloaded, and may not contain up-to-date parameters
    """
    parser = get_parser()
    # get all defaults (simpler for debugging)
    defaults = {}
    for action in parser._actions:  # pylint: disable=protected-access
        if not action.required and action.dest != "help":
            defaults[action.dest] = action.default
    if isinstance(parameters, AttrDict):
        for p, val in defaults.items():
            if p not in parameters.__dict__:
                if log:
                    logger.info("Adding default value %s for %s in parameter", val, p)
                parameters.__dict__[p] = val


def set_pretrain_emb(model, dico, word2id, embeddings, gpu):
    """
    Pretrain word embeddings.
    """
    n_found = 0
    with torch.no_grad():
        for i in range(len(dico)):
            idx = word2id.get(dico[i], None)
            if idx is None:
                continue
            n_found += 1
            model.embeddings.weight[i] = (
                embeddings[idx].cuda() if gpu else embeddings[idx]
            )
            model.pred_layer.proj.weight[i] = (
                embeddings[idx].cuda() if gpu else embeddings[idx]
            )
    logger.info(
        "Pretrained %i/%i words (%.3f%%)."
        % (n_found, len(dico), 100.0 * n_found / len(dico))
    )


@torch.no_grad()
def build_model(params, dico, gpu=True):
    """
        Build model.
    """
    add_missing_parameters(params)

    if params.encoder_only:
        # build
        model = TransformerModel(params, dico, is_encoder=True, with_output=True)

        # reload pretrained word embeddings
        if params.reload_emb != "":
            word2id, embeddings = load_embeddings(params.reload_emb, params)
            set_pretrain_emb(model, dico, word2id, embeddings, gpu)

        # reload a pretrained model
        if params.reload_model != "":
            logger.info("============ Model Reloading")
            logger.info("Reloading model from %s ..." % params.reload_model)
            reload_transformer(
                params, params.reload_model, dico, model, "model", gpu=gpu
            )

        logger.info("Model: {}".format(model))
        logger.info(
            "Number of parameters (model): %i"
            % sum([p.numel() for p in model.parameters() if p.requires_grad])
        )
        logger.info("")

        return [model.cuda() if gpu else model]

    else:
        # build
        # TODO: only output when necessary - len(params.clm_steps + params.mlm_steps) > 0
        encoder = TransformerModel(params, dico, is_encoder=True, with_output=True)

        if params.separate_decoders:
            decoders = [
                TransformerModel(params, dico, is_encoder=False, with_output=True)
                for _ in params.lang2id.values()
            ]
        else:
            decoders = [
                TransformerModel(params, dico, is_encoder=False, with_output=True)
            ]

        for layer in range(params.n_layers_decoder):
            if layer <= params.n_share_dec - 1:
                assert params.amp == -1, "sharing layers is not supported with AMP"
                logger.info("Sharing decoder attention parameters for layer %i" % layer)
                for i in range(1, len(decoders)):
                    decoders[i].attentions[layer] = decoders[0].attentions[layer]

        # reload pretrained word embeddings
        if params.reload_emb != "":
            word2id, embeddings = load_embeddings(params.reload_emb, params)
            set_pretrain_emb(encoder, dico, word2id, embeddings, gpu)
            for decoder in decoders:
                set_pretrain_emb(decoder, dico, word2id, embeddings, gpu)

        # reload a pretrained model
        if params.reload_model != "":
            logger.info("============ Model Reloading")
            enc_path, dec_path = params.reload_model.split(",")
            assert not (enc_path == "" and dec_path == "")

            # reload encoder
            if enc_path != "":
                logger.info("Reloading encoder from %s ..." % enc_path)
                reload_transformer(params, enc_path, dico, encoder, "encoder", gpu=gpu)

            # reload decoders
            if dec_path != "":
                for i, dec in enumerate(decoders):
                    logger.info("Reloading decoders from %s ..." % dec_path)
                    if params.reload_encoder_for_decoder:
                        reload_transformer(
                            params, dec_path, dico, dec, "encoder", gpu=gpu
                        )
                    else:
                        reload_transformer(
                            params, dec_path, dico, dec, "decoder", gpu, i
                        )

        logger.debug("Encoder: {}".format(encoder))
        logger.debug("Decoder: {}".format(decoders))
        logger.info(
            "Number of parameters (encoder): %i"
            % sum([p.numel() for p in encoder.parameters() if p.requires_grad])
        )
        logger.info(
            "Number of parameters (decoders): %i"
            % sum([p.numel() for p in decoders[0].parameters() if p.requires_grad])
        )
        logger.info(f"Number of decoders: {len(decoders)}")
        logger.info("")

        return (
            [encoder.cuda() if gpu else encoder],
            [dec.cuda() if gpu else dec for dec in decoders],
        )


def reload_transformer(params, path, dico, model, model_type, gpu=True, model_number=None):
    """
    Reload a transformer state dict to current model:
    clean 'module.' from state dict,
    match the word embeddings comparing dicos,
    match lang embedding with params lang mapping,
    extend or truncate position embeddings when size don't match,
    load state dict.
    """

    reloaded = torch.load(
        path,
        map_location=lambda storage, loc: storage.cuda(params.local_rank)
        if gpu
        else storage.cpu(),
    )
    if "state_dicts" in reloaded:  # compatibility with new online pipeline
        logger.warning("Reloading from multi checkpoint (skipping safety checks)")
        for name in ["encoder", "decoder"]:
            reloaded[name] = reloaded["state_dicts"]["models/" + name]
        pdict = {f.name: getattr(params, f.name) for f in dataclasses.fields(params)}
        reloaded["params"] = pdict
        word2id = reloaded.get("word2id", None)
        if word2id is None:
            logger.warning(
                "word2id is missing in reloaded checkpoint, assuming current ones"
            )
            word2id = model.dico
        reloaded["dico_word2id"] = word2id
        reloaded["dico_id2word"] = {y: x for x, y in word2id.items()}
    clean_model_state_dict(reloaded, model_type, model_number)
    reload_word_embeddings(reloaded, dico, model_type)
    reload_lang_embeddings(reloaded, params, model_type)
    reload_position_embeddings(reloaded, model, model_type)

    # if the model is a decoder
    if hasattr(model, "encoder_attn"):
        for i in range(params.n_layers_decoder):
            for name in DECODER_ONLY_PARAMS:
                weight_name = name % i
                if weight_name not in reloaded[model_type]:
                    logger.warning("Parameter %s not found." % (weight_name))
                    encoder_attn_name = weight_name.replace(
                        "encoder_attn", "attentions"
                    )
                    if (
                        getattr(params, "reload_encoder_attn_on_decoder", False)
                        and "encoder_attn" in weight_name
                        and encoder_attn_name in reloaded[model_type]
                    ):
                        logger.warning(f"Reloading {encoder_attn_name} instead")
                        reloaded[model_type][weight_name] = (
                            reloaded[model_type][encoder_attn_name].clone().detach()
                        )
                    else:
                        reloaded[model_type][weight_name] = model.state_dict()[
                            weight_name
                        ]
    model.load_state_dict(reloaded[model_type], strict=not params.spans_emb_encoder)


def clean_model_state_dict(reloaded, model_type, model_number=None):
    """
    remove prefix module from the keys of the model state dict.
    """

    type_with_number = f"{model_type}_{model_number}"
    if model_number is not None and type_with_number in reloaded:
        model_reloaded = reloaded[type_with_number]
    else:
        if model_number is not None:
            logger.info(
                f"{type_with_number} not in reloaded model, reloading {model_type}"
            )
        model_reloaded = reloaded[model_type if model_type in reloaded else "model"]

    if all([k.startswith("module.") for k in model_reloaded.keys()]):
        model_reloaded = {k[len("module.") :]: v for k, v in model_reloaded.items()}
    reloaded[model_type] = model_reloaded


def reload_word_embeddings(reloaded, dico, model_type):
    """
    Check when reloading a model that dictionary are the same. If not, do a word embedding mapping if possible.
    """
    reloaded_word2id = reloaded["dico_word2id"]
    reloaded_id2word = reloaded["dico_id2word"]
    assert len(reloaded_word2id) == len(reloaded_id2word)
    assert all(reloaded_id2word[v] == k for k, v in reloaded_word2id.items())

    matching_indices = []
    word_not_found = []
    for idx, word in dico.id2word.items():
        if word not in reloaded_word2id:
            word_not_found += [word]
            matching_indices += [reloaded_word2id[UNK_WORD]]
        else:
            matching_indices += [reloaded_word2id[word]]
    assert len(matching_indices) == len(dico)
    if len(word_not_found) > 0:
        logger.warning(
            f"When reloading word embeddings, could not find embeddings for {len(word_not_found)} words: {word_not_found[0:5] + ['...'] + word_not_found[-5:]}... Initializing them to < unk >."
        )

    reloaded[model_type]["embeddings.weight"] = torch.cat(
        [
            reloaded[model_type]["embeddings.weight"][index : index + 1]
            for index in matching_indices
        ],
        dim=0,
    )

    if "pred_layer.proj.weight" in reloaded[model_type]:
        first_line = reloaded[model_type]["pred_layer.proj.weight"][0:1]
        embedding_size = reloaded[model_type]["pred_layer.proj.weight"].shape[1]
        reloaded[model_type]["pred_layer.proj.weight"] = torch.cat(
            [
                reloaded[model_type]["pred_layer.proj.weight"][index : index + 1]
                if index is not None
                else torch.normal(
                    torch.zeros_like(first_line),
                    torch.ones_like(first_line * (embedding_size ** (-0.5))),
                )
                for index in matching_indices
            ],
            dim=0,
        )
        reloaded[model_type]["pred_layer.proj.bias"] = torch.cat(
            [
                reloaded[model_type]["pred_layer.proj.bias"][index].view(1)
                if index is not None
                else torch.rand_like(
                    reloaded[model_type]["pred_layer.proj.bias"][0].view(1)
                )
                for index in matching_indices
            ]
        )


def reload_lang_embeddings(reloaded, params, model_type):
    """
    When pretrained models has not been trained with the same languages:
    change lang embedding state dict.
    Otherwise, keep as it is.
    """
    model_reloaded = reloaded[model_type]
    reloaded_params = reloaded["params"]
    if params.lgs_mapping == "":
        lang_mapping = {}
    else:
        lang_mapping = {
            mapping.split(":")[0]: mapping.split(":")[1]
            for mapping in params.lgs_mapping.split(",")
        }
    langs_reloaded = reloaded_params["lang2id"]
    langs_reloaded_id2lang = reloaded_params["id2lang"]
    indices = []
    for lang in [l for i, l in sorted(params.id2lang.items())]:
        if lang in lang_mapping:
            lang_ = lang_mapping[lang]
        else:
            lang_ = lang
        index = [id for l, id in langs_reloaded.items() if l == lang_]
        if len(index) == 0:
            logger.warning(
                f"No match found for lang {lang} {lang_} in {langs_reloaded.keys()}. Initializing randomly."
            )
            indices.append(None)
            continue
        else:
            assert (
                len(index) == 1
            ), f"matching lang found: {index} in reloaded model for lang {lang} in {langs_reloaded.keys()}"
            logger.warning(
                f"Lang {lang} matched to pretrained {langs_reloaded_id2lang[index[0]]} lang embedding."
            )
        indices.append(index[0])

    first_line = model_reloaded["lang_embeddings.weight"][0:1]
    embedding_size = model_reloaded["lang_embeddings.weight"].shape[1]
    model_reloaded["lang_embeddings.weight"] = torch.cat(
        [
            model_reloaded["lang_embeddings.weight"][index : index + 1]
            if index is not None
            else torch.normal(
                torch.zeros_like(first_line),
                torch.ones_like(first_line * (embedding_size ** (-0.5))),
            )
            for index in indices
        ],
        dim=0,
    )
    reloaded[model_type] = model_reloaded


def reload_position_embeddings(reloaded, encoder, model_type):
    """
    When pretrained models has not been trained with the same size of position embedding:
    remove unused or add extra positions.
    """
    model_reloaded = reloaded[model_type]
    current_size = encoder.position_embeddings.weight.size()[0]
    reloaded_size = model_reloaded["position_embeddings.weight"].size()[0]
    if current_size == reloaded_size:
        return model_reloaded
    elif current_size < reloaded_size:
        logger.warning(
            f"The size of position embeddings in current model is {current_size}, the size of reloaded is {reloaded_size}. need to truncate the reloaded position embeddings."
        )
        model_reloaded["position_embeddings.weight"] = model_reloaded[
            "position_embeddings.weight"
        ][:current_size, :]
    else:
        logger.warning(
            f"The size of position embeddings in current model is {current_size}, the size of reloaded is {reloaded_size}. need to repeat last positions {current_size - reloaded_size} times."
        )
        model_reloaded["position_embeddings.weight"] = torch.cat(
            [
                model_reloaded["position_embeddings.weight"],
                model_reloaded["position_embeddings.weight"][-1, :].repeat(
                    current_size - reloaded_size, 1
                ),
            ],
            dim=0,
        )
    reloaded[model_type] = model_reloaded
