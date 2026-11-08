# MuSiQue development subset

MuSiQue: Multi-hop Questions via Single-hop Question Composition, by Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot and Ashish Sabharwal, TACL 2022. The [official repository](https://github.com/StonyBrookNLP/musique) distributes MuSiQue under **CC BY 4.0**. This attribution/license applies to the dataset; it does not license model weights or project code.

This derivative contains 96 questions from MuSiQue-Ans v1.0 development: 32 per 2-, 3- and 4-hop stratum, selected by fixed question-ID hash order without model inference. Text and answer aliases are preserved; paragraph support flags are converted to source IDs. Question decompositions are not model inputs. `prepare_v4_musique.py` records the transformations and raw/output hashes.

The downloaded source is the [dgslibisey/MuSiQue HF mirror](https://huggingface.co/datasets/dgslibisey/MuSiQue), pinned to commit `c8f4f8c9465fb69d31a8eae894c3fd509c4ca321`. Its filenames/schema correspond to the official v1.0 release; byte equivalence to the official Google Drive ZIP has **not** been independently verified. This limitation is recorded in the manifest. The subset is not human-reaudited, and is exploratory development rather than final test evidence. Gold support annotations are used only for reporting and an explicitly labeled oracle ceiling.
