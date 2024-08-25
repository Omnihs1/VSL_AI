# VSL_AI
A graph model for Vietnamese Sign Language
## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation
- Create a new conda environment with required package
```bash
$ conda create --name <env> --file requirements.txt
```
## Usage
- **Step 1 : Extract keypoints** <br>
Using csv file like "template/template_file.csv" as input 
```python
python extract_keypoints.py
```
- **Step 2 : Preprocessing** <br>
Using output csv file in Step 1 in line 48 of pre_processing.py and replace path variable to what you use
```python
python pre_processsing.py
```
The results are npy file like data folder of repo
- **Step 3 : Train model** <br>
Replace npy path from line 41 to 44 in train_vsl.py
```python
python train_vsl.py
```
- **Step 4 : Test model** <br>
Comment code in line 58, uncomment code in line 60 to 61, replace model path
```python
python train_vsl.py
```
## License
This project is licensed under the [MIT License](LICENSE).

MIT License

Copyright (c) [2024] [Son Ha]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.