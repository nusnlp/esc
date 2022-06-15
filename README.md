# Frustratingly Easy System Combination for Grammatical Error Correction

This repository provides the code to easily combines Grammatical Error Correction (GEC) models to produce better predictions with just the models' outputs, as reported in this paper:

> Frustratingly Easy System Combination for Grammatical Error Correction <br>
> [Muhammad Reza Qorib](https://mrqorib.github.io/), [Seung-Hoon Na](https://nlp.jbnu.ac.kr/~nash/faculty.html), and [Hwee Tou Ng](https://www.comp.nus.edu.sg/~nght/) <br>
> 2022 Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL) ([PDF](https://mrqorib.github.io/assets/pdf/ESC.pdf))

## Installation
This code should be run with Python 3.6. The reason Python 3.6 is needed is because the ERRANT version that is used in the BEA-2019 shared task (v2.0.0) is not compatible with Python >= 3.7

Install this code dependencies by running:
```.bash
pip install -r requirements.txt
python -m spacy download en
wget https://www.comp.nus.edu.sg/~nlp/sw/m2scorer.tar.gz
tar -xf m2scorer.tar.gz
```
Note that you may need to customize your pytorch installation depending on your CUDA version, read more [here](https://pytorch.org/get-started/previous-versions/). The code may also work with torch < 1.9.0 as only simple pytorch functions are used.

## Reproducing the paper's result
For the CoNLL-2014 experiment, run: `export EXP_DIR=conll-exp` .

For the BEA-2019 experiment, run: `export EXP_DIR=bea-exp` .
1. Get the model output
```.bash
python run.py --test --data_dir $EXP_DIR/test-text --m2_dir $EXP_DIR/test-m2 --model_path $EXP_DIR/models/paper_model.pt --vocab_path $EXP_DIR/paper_vocab.idx --output_path $EXP_DIR/outputs/test.out
```
2. [Evaluate](#evaluation) the test prediction. Replace test_output with $EXP_DIR/outputs/test.out

## Retraining the experiments in the paper
For the CoNLL-2014 experiment, run: `export EXP_DIR=conll-exp` .

For the BEA-2019 experiment, run: `export EXP_DIR=bea-exp` .
1. Run the training command: 
```.bash
python run.py --train --data_dir $EXP_DIR/dev-text --m2_dir $EXP_DIR/dev-m2 --model_path $EXP_DIR/models --vocab_path $EXP_DIR/vocab.idx
```
2. Get the prediction on BEA-2019 Dev:
```.bash
python run.py --test --data_dir $EXP_DIR/dev-text --m2_dir $EXP_DIR/dev-m2 --model_path $EXP_DIR/models/model.pt --vocab_path $EXP_DIR/vocab.idx --output_path $EXP_DIR/outputs/dev.out
```
3. Get the F0.5 development score:
```.bash
errant_parallel -ori $EXP_DIR/dev-text/source.txt -cor $EXP_DIR/outputs/dev.out -out $EXP_DIR/outputs/dev.m2
errant_compare -ref bea-full-valid.m2 -hyp $EXP_DIR/outputs/dev.m2
```
4. Get the test prediction:
```.bash
python run.py --test --data_dir $EXP_DIR/test-text --m2_dir $EXP_DIR/test-m2 --model_path $EXP_DIR/models/model.pt --vocab_path $EXP_DIR/vocab.idx --output_path $EXP_DIR/outputs/test.out
```
7. [Evaluate](#evaluation) the test prediction. Replace test_output with $EXP_DIR/outputs/test.out

## Evaluation
- For CoNLL-2014 (requires Python 2.x):
```.bash
python2 m2scorer/scripts/m2scorer.py test_output conll14st-test-corrected.m2
```
- For BEA-2019:
Compress the `test_output` into test.zip, then upload the zip file (only containing the prediction file without any folder or any other file) to https://competitions.codalab.org/competitions/20228#participate-get-data

## Combining your own systems
The simplest way is:
- Create a new experiment directory, then go inside this directory.
- Put your base systems' output on BEA-2019 Dev in a folder called `dev-text`. Please also copy the `source.txt` and `target.txt` from the `bea-exp/dev-text` folder to this new `dev-text` folder.
- Put your base system's output on the test set in a folder called `test-text`. Please also put the source sentences of the dataset you are testing with inside the folder, under the name of `source.txt`.
- Create the `models` and `outputs` folder. At this point, make sure your folder structure is similar to the contents of `bea-exp` or `conll-exp`, with the exceptions of `dev-m2` and `test-m2` (The code will generate these folders automatically). 
- Go back to the parent directory and follow the [guide](#retraining-the-experiments-in-the-paper) above, with the $EXP_DIR replaced with your new folder name.

If you want to customize your experiment setup, please note:
- The code will index all files in the `--data_dir` folder as base systems, except the source file (the default filename is `source.txt`) and the target file (the default filename is `target.txt`).
- The code will only read the contents of `--m2_dir`, not `--data_dir`.  The code will index the files in `--data_dir` and look for the file with same basename on the `--m2_dir`.If the `--m2_dir` does not exist, the code will generate the directory along with the contents from the content of `--data_dir`. Thus, if you make any changes to the content of `--data_dir` after `--m2_dir` was generated, please remove the corresponding file on the `--m2_dir` or the delete the whole `--m2_dir` entirely.
- The file names of the training files and the testing files have to be the same. The file names and the ordering are stored in the vocab file.
- When you run the testing, make sure you run the prediction with the correct model and correct vocab file. Both files are dependent to the base systems you are combining.

## License
The source code and models in this repository are licensed under the GNU General Public License Version 3 (see [License](./LICENSE.txt)). For commercial use of this code and models, separate commercial licensing is also available. Please contact Hwee Tou Ng (nght@comp.nus.edu.sg)
