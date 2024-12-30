# Worker Models Directory

This folder is the default location and can be control by setting `cache_home` in the config or by setting the `AIWORKER_CACHE_HOME` environment variable.

## clip_blip

The interrogation (tag similiarity) and captioning models used by alchemy workers and for the safety process in image workers.

## codeformer

The codeformer series of facefixers used by the alchemy and during image post-processing on image workers.

## compvis

An obsolete name, originally due to Stable Diffusion being made by compvis, this folder now actually contains image generation models of *all* types, including SDXL (by stability ai) and Flux (by black forest labs).

## controlnet

The controlnet models used by image workers. See the original github repo for controlnets for a more detailed explanation of the function of these models [here](https://github.com/lllyasviel/ControlNet).

## custom

Any custom models being offered by worker operators with the `customizer` role. See the main readme for more information.

## esrgan

The esrgan series of upscalers used by alchemy and during post-processing by image workers.

## gfpgan

The gfpgan series of face fixers used by alchemy and during post-processing by image workers.

## horde_model_reference

A number of json files fetched from the [official repositories](https://github.com/Haidra-Org/AI-Horde-image-model-reference) which determine what model versions the horde allows. These files include integrity information (hashes) that the worker uses to ensure the models it downloads are the right ones.

## lora

Only used if `allow_loras: true` is set in your config. The lora models downloaded from the third party to fufil requests which use loras. The worker only downloads `.safetensors` loras for your safety.

## miscellaneous

A collection of other models which are part of advanced features the horde worker supports.

## safety_checker

Obsolete.

## ti

The embedding models (Textual Inversions, "TIs") downloaded as needed when a request requires them. These files are provided by the horde as `.safetensors` to ensure your machine's safety.
