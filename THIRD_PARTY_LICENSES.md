<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
-->

# Third-Party Notices

This project (Spatial-IQ) is distributed under the BSD-3-Clause license
(see [LICENSE](LICENSE)). It depends on the following third-party open-source
packages at runtime. None of these packages are redistributed inside this
repository — they are installed by the end user via `pip`, `conda`, or the
NVIDIA Isaac Sim installer — but their notices are reproduced here for
attribution.

The dependency lists come from the following manifests:

- `analyses/requirements.txt`
- `evaluation/requirements.txt`
- `evaluation/main_ocr/requirements.txt`
- `inference/gemini/requirements.txt`
- `inference/qwen/requirements_image_edit.txt`
- `training_evaluation/requirements.txt` (plus VLMEvalKit itself, cloned by `training_evaluation/setup.sh`)
- `pyproject.toml` (docs site build)

## Overview by category

### Core scientific stack (analyses, data_generation, evaluation)

- NumPy — BSD-3-Clause — numerical arrays
- SciPy — BSD-3-Clause — statistics
- pandas — BSD-3-Clause — table / CSV I/O
- Matplotlib — Matplotlib License (PSF-based, BSD-compatible) — plotting
- Pillow — HPND (PIL Software License) — image I/O
- tqdm — MIT + MPL-2.0 (dual) — progress bars
- requests — Apache-2.0 — HTTP client used by API inference clients

### Storage and I/O

- boto3 / botocore — Apache-2.0 — used by `data_generation/blocks_render_image.py` only when `UPLOAD_TO_S3=1` (optional)

### API / OpenAI client (dual use)

- openai (Python client library) — Apache-2.0

Used in two distinct contexts:
1. **`evaluation/main_ocr/`** — calls a locally hosted, OpenAI-compatible vLLM endpoint. No OpenAI account or API key required.
2. **`training_evaluation/` (via VLMEvalKit)** — calls the OpenAI API to power MathVista's GPT-graded answer extraction. Requires the user to supply their own `OPENAI_API_KEY` at runtime. Never bundled or logged.

### OCR (`evaluation/main_ocr/`)

- PaddleOCR — Apache-2.0
- PaddlePaddle — Apache-2.0
- pytesseract — Apache-2.0
- Tesseract OCR — Apache-2.0 (system binary, installed via conda-forge)

### Deep-learning framework and model loading (`inference/qwen/…image_edit`, `training_evaluation/`)

- PyTorch (`torch`, `torchvision`) — modified BSD (PyTorch License, BSD-3-Clause style)
- Hugging Face Transformers — Apache-2.0
- Hugging Face Accelerate — Apache-2.0
- Hugging Face Hub — Apache-2.0
- Hugging Face Datasets — Apache-2.0
- Hugging Face Diffusers — Apache-2.0
- SentencePiece — Apache-2.0
- tiktoken — MIT
- timm — Apache-2.0
- einops — MIT
- protobuf — BSD-3-Clause
- qwen-vl-utils — Apache-2.0
- typing_extensions — PSF (Python Software Foundation License, BSD-compatible)

### VLMEvalKit itself and its wider runtime (`training_evaluation/`)

- VLMEvalKit — Apache-2.0 — cloned from `https://github.com/open-compass/VLMEvalKit.git` by `training_evaluation/setup.sh`. Installed with `--no-deps` and complemented by the pinned dependency set in `training_evaluation/requirements.txt`.
- openpyxl — MIT
- XlsxWriter — BSD-2-Clause
- rich — MIT
- tabulate — MIT
- termcolor — MIT
- portalocker — BSD-3-Clause
- sty — MIT
- python-dotenv — BSD-3-Clause
- json_repair — MIT
- nest_asyncio — BSD-2-Clause
- NLTK — Apache-2.0
- rouge — Apache-2.0
- scikit-learn — BSD-3-Clause
- scikit-image — BSD-3-Clause
- bert_score — MIT
- editdistance — MIT
- Levenshtein — MIT (the `Levenshtein` PyPI distribution; not the historical GPL-licensed `python-Levenshtein`)
- SymPy — BSD-3-Clause
- pylatexenc — MIT
- **num2words — LGPL-2.1** — runtime library only; not modified, not statically linked. See notes below.
- jieba — MIT
- anls — MIT
- distance — MIT
- zss — MIT
- apted — MIT
- lxml — BSD-3-Clause
- opencv-python — Apache-2.0
- decord — Apache-2.0
- imageio — BSD-2-Clause
- validators — MIT

### Documentation site (`pyproject.toml`, docs/)

- Sphinx — BSD-2-Clause
- NVIDIA Sphinx Theme — Apache-2.0
- MyST-Parser — MIT

### Native NVIDIA / simulator

- NVIDIA Isaac Sim (Kit, USD, Replicator) — governed by the NVIDIA Isaac Sim EULA. Installed separately by the end user; only public Python APIs (`omni.*`, `carb.*`, `pxr`) are imported.

## License-family notes

**No GPL or AGPL dependencies.** The only copyleft dependency is `num2words` (LGPL-2.1), pulled in transitively via VLMEvalKit for spelling out numeric answers during MathVista scoring. Spatial-IQ does not modify or statically link `num2words`; it is invoked at runtime through its public Python API, which is the case explicitly permitted by LGPL. If your downstream distribution has a stricter no-copyleft policy, you can safely remove `num2words` from `training_evaluation/requirements.txt` — VLMEvalKit falls back gracefully when the package is unavailable.

---

## NumPy

- Homepage: https://numpy.org/
- Source:   https://github.com/numpy/numpy
- License:  BSD-3-Clause
- License text: https://github.com/numpy/numpy/blob/main/LICENSE.txt

> Copyright (c) 2005-2024, NumPy Developers.
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are
> met:
>
>     * Redistributions of source code must retain the above copyright
>        notice, this list of conditions and the following disclaimer.
>
>     * Redistributions in binary form must reproduce the above
>        copyright notice, this list of conditions and the following
>        disclaimer in the documentation and/or other materials provided
>        with the distribution.
>
>     * Neither the name of the NumPy Developers nor the names of any
>        contributors may be used to endorse or promote products derived
>        from this software without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
> "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
> LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
> A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
> OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
> SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
> LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
> DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
> THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
> (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
> OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

---

## SciPy

- Homepage: https://scipy.org/
- Source:   https://github.com/scipy/scipy
- License:  BSD-3-Clause
- License text: https://github.com/scipy/scipy/blob/main/LICENSE.txt

> Copyright (c) 2001-2002 Enthought, Inc. 2003-2024, SciPy Developers.
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions
> are met:
>
> 1. Redistributions of source code must retain the above copyright
>    notice, this list of conditions and the following disclaimer.
>
> 2. Redistributions in binary form must reproduce the above
>    copyright notice, this list of conditions and the following
>    disclaimer in the documentation and/or other materials provided
>    with the distribution.
>
> 3. Neither the name of the copyright holder nor the names of its
>    contributors may be used to endorse or promote products derived
>    from this software without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
> "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
> LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
> A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
> HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
> SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
> LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
> DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
> THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
> (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
> OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

---

## pandas

- Homepage: https://pandas.pydata.org/
- Source:   https://github.com/pandas-dev/pandas
- License:  BSD-3-Clause
- License text: https://github.com/pandas-dev/pandas/blob/main/LICENSE

> BSD 3-Clause License
>
> Copyright (c) 2008-2011, AQR Capital Management, LLC, Lambda Foundry, Inc.
> and PyData Development Team
> All rights reserved.
>
> Copyright (c) 2011-2024, Open source contributors.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> * Redistributions of source code must retain the above copyright notice, this
>   list of conditions and the following disclaimer.
>
> * Redistributions in binary form must reproduce the above copyright notice,
>   this list of conditions and the following disclaimer in the documentation
>   and/or other materials provided with the distribution.
>
> * Neither the name of the copyright holder nor the names of its
>   contributors may be used to endorse or promote products derived from
>   this software without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
> DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
> FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
> DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
> SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
> CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
> OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
> OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

---

## Matplotlib

- Homepage: https://matplotlib.org/
- Source:   https://github.com/matplotlib/matplotlib
- License:  Matplotlib License (PSF-based, BSD-compatible)
- License text: https://github.com/matplotlib/matplotlib/blob/main/LICENSE/LICENSE

> License agreement for matplotlib versions 1.3.0 and later
> =========================================================
>
> 1. This LICENSE AGREEMENT is between the Matplotlib Development Team
> ("MDT"), and the Individual or Organization ("Licensee") accessing and
> otherwise using matplotlib software in source or binary form and its
> associated documentation.
>
> 2. Subject to the terms and conditions of this License Agreement, MDT
> hereby grants Licensee a nonexclusive, royalty-free, world-wide license
> to reproduce, analyze, test, perform and/or display publicly, prepare
> derivative works, distribute, and otherwise use matplotlib
> alone or in any derivative version, provided, however, that MDT's
> License Agreement and MDT's notice of copyright, i.e., "Copyright (c)
> 2012- Matplotlib Development Team; All Rights Reserved" are retained in
> matplotlib alone or in any derivative version prepared by
> Licensee.
>
> 3. In the event Licensee prepares a derivative work that is based on or
> incorporates matplotlib or any part thereof, and wants to
> make the derivative work available to others as provided herein, then
> Licensee hereby agrees to include in any such work a brief summary of
> the changes made to matplotlib.
>
> 4. MDT is making matplotlib available to Licensee on an "AS
> IS" basis. MDT MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR
> IMPLIED. BY WAY OF EXAMPLE, BUT NOT LIMITATION, MDT MAKES NO AND
> DISCLAIMS ANY REPRESENTATION OR WARRANTY OF MERCHANTABILITY OR FITNESS
> FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF MATPLOTLIB
> WILL NOT INFRINGE ANY THIRD PARTY RIGHTS.
>
> 5. MDT SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF MATPLOTLIB
> FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR LOSS AS
> A RESULT OF MODIFYING, DISTRIBUTING, OR OTHERWISE USING MATPLOTLIB,
> OR ANY DERIVATIVE THEREOF, EVEN IF ADVISED OF THE POSSIBILITY THEREOF.
>
> 6. This License Agreement will automatically terminate upon a material
> breach of its terms and conditions.
>
> 7. Nothing in this License Agreement shall be deemed to create any
> relationship of agency, partnership, or joint venture between MDT and
> Licensee. This License Agreement does not grant permission to use MDT
> trademarks or trade name in a trademark sense to endorse or promote
> products or services of Licensee, or any third party.
>
> 8. By copying, installing or otherwise using matplotlib,
> Licensee agrees to be bound by the terms and conditions of this License
> Agreement.

---

## Pillow

- Homepage: https://python-pillow.org/
- Source:   https://github.com/python-pillow/Pillow
- License:  HPND (Historical Permission Notice and Disclaimer — PIL Software License, BSD-compatible)
- License text: https://github.com/python-pillow/Pillow/blob/main/LICENSE

> The Python Imaging Library (PIL) is
>
>     Copyright © 1997-2011 by Secret Labs AB
>     Copyright © 1995-2011 by Fredrik Lundh and Contributors
>
> Pillow is the friendly PIL fork. It is
>
>     Copyright © 2010 by Jeffrey A. Clark and contributors
>
> Like PIL, Pillow is licensed under the open source HPND License:
>
> By obtaining, using, and/or copying this software and/or its associated
> documentation, you agree that you have read, understood, and will comply
> with the following terms and conditions:
>
> Permission to use, copy, modify and distribute this software and its
> documentation for any purpose and without fee is hereby granted,
> provided that the above copyright notice appears in all copies, and that
> both that copyright notice and this permission notice appear in supporting
> documentation, and that the name of Secret Labs AB or the author not be
> used in advertising or publicity pertaining to distribution of the software
> without specific, written prior permission.
>
> SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
> SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
> IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR BE LIABLE FOR ANY SPECIAL,
> INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
> LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
> OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
> PERFORMANCE OF THIS SOFTWARE.

---

## tqdm

- Homepage: https://tqdm.github.io/
- Source:   https://github.com/tqdm/tqdm
- License:  MIT + MPL 2.0 (dual-licensed)
- License text: https://github.com/tqdm/tqdm/blob/master/LICENCE

> `tqdm` is a product of collaborative work.
>
> Unless otherwise stated, all authors (see commit logs) retain copyright
> for their respective work, and release the work under the MIT licence
> (text below).
>
> Exceptions or notable authors are listed [in the source LICENCE file].
>
> --- MIT License ---
>
> Copyright (c) 2013 noamraph
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in
> all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

---

## boto3 / botocore

- Homepage: https://boto3.amazonaws.com/
- Source:   https://github.com/boto/boto3 · https://github.com/boto/botocore
- License:  Apache-2.0
- License text: https://github.com/boto/boto3/blob/develop/LICENSE

> Copyright 2013-2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
>
> Licensed under the Apache License, Version 2.0 (the "License"). You may not
> use this file except in compliance with the License. A copy of the License
> is located at
>
>     http://aws.amazon.com/apache2.0/
>
> or in the "license" file accompanying this file. This file is distributed on
> an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
> express or implied. See the License for the specific language governing
> permissions and limitations under the License.

---

## openai (Python client library)

- Homepage: https://github.com/openai/openai-python
- Source:   https://github.com/openai/openai-python
- License:  Apache-2.0
- License text: https://github.com/openai/openai-python/blob/main/LICENSE

> Copyright 2024 OpenAI
>
> Licensed under the Apache License, Version 2.0 (the "License"); you may not
> use this file except in compliance with the License. You may obtain a copy of
> the License at
>
>     http://www.apache.org/licenses/LICENSE-2.0
>
> Unless required by applicable law or agreed to in writing, software
> distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
> WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
> License for the specific language governing permissions and limitations
> under the License.

*Note on usage: this project uses the `openai` client library in two distinct
contexts.*

*1. **`evaluation/main_ocr/`** invokes a locally hosted, OpenAI-compatible vLLM
endpoint; no OpenAI account or API key is required for this path.*

*2. **`training_evaluation/` (via VLMEvalKit)** calls the OpenAI API to power
MathVista's GPT-graded answer extraction; this path requires the user to supply
their own `OPENAI_API_KEY` at runtime. No OpenAI key is bundled with the
repository, and none is logged or persisted.*

---

## PaddleOCR

- Homepage: https://github.com/PaddlePaddle/PaddleOCR
- Source:   https://github.com/PaddlePaddle/PaddleOCR
- License:  Apache-2.0
- License text: https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE

> Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
>
> Licensed under the Apache License, Version 2.0 (the "License"); you may not
> use this file except in compliance with the License. You may obtain a copy
> of the License at
>
>     http://www.apache.org/licenses/LICENSE-2.0
>
> Unless required by applicable law or agreed to in writing, software
> distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
> WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
> License for the specific language governing permissions and limitations
> under the License.

---

## PaddlePaddle

- Homepage: https://www.paddlepaddle.org/
- Source:   https://github.com/PaddlePaddle/Paddle
- License:  Apache-2.0
- License text: https://github.com/PaddlePaddle/Paddle/blob/develop/LICENSE

> Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
>
> Licensed under the Apache License, Version 2.0 (the "License"); you may not
> use this file except in compliance with the License. You may obtain a copy
> of the License at
>
>     http://www.apache.org/licenses/LICENSE-2.0
>
> Unless required by applicable law or agreed to in writing, software
> distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
> WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

---

## pytesseract

- Homepage: https://github.com/madmaze/pytesseract
- Source:   https://github.com/madmaze/pytesseract
- License:  Apache-2.0
- License text: https://github.com/madmaze/pytesseract/blob/master/LICENSE

> Copyright 2011-2024 Matthias A Lee
>
> Licensed under the Apache License, Version 2.0 (the "License"); you may not
> use this file except in compliance with the License. You may obtain a copy
> of the License at
>
>     http://www.apache.org/licenses/LICENSE-2.0
>
> Unless required by applicable law or agreed to in writing, software
> distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
> WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

---

## Tesseract OCR

- Homepage: https://github.com/tesseract-ocr/tesseract
- Source:   https://github.com/tesseract-ocr/tesseract
- License:  Apache-2.0
- License text: https://github.com/tesseract-ocr/tesseract/blob/main/LICENSE

> Copyright (c) 1985, 1994, 1995 IBM Corporation. All rights reserved.
> Copyright (c) 2007-2024 The Tesseract Contributors. All Rights Reserved.
>
> Licensed under the Apache License, Version 2.0 (the "License"); you may not
> use this file except in compliance with the License. You may obtain a copy
> of the License at
>
>     http://www.apache.org/licenses/LICENSE-2.0
>
> Unless required by applicable law or agreed to in writing, software
> distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
> WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

*Note: the Tesseract binary is installed via conda-forge as part of the
`evaluation/main_ocr` OCR toolchain and is not redistributed by this
repository.*

---

## NVIDIA Isaac Sim (Kit, USD, Replicator)

- Homepage: https://developer.nvidia.com/isaac-sim
- License:  Governed by the [NVIDIA Isaac Sim EULA](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/guide/end_user_license_agreement.html) — free to use for authorized users.

The Isaac Sim data-generation code (`data_generation/blocks_render_image.py`)
imports the public Python APIs `omni.*`, `carb.*`, and `pxr` provided by
Isaac Sim. **Isaac Sim itself is not redistributed by this repository**; end
users install Isaac Sim 4.5 separately under NVIDIA's Isaac Sim EULA. The USD
library (`pxr`) that Isaac Sim ships is a modified Apache-2.0 build; see the
NVIDIA Isaac Sim license notices for details.
