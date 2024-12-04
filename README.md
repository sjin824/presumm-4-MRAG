# presumm-4-MRAG

This project is a customized version of [EMNLP 2019: PreSumm](https://github.com/nlpyang/PreSumm), designed specifically for integration with [MRAG]() (currently unavailable).
Below, we outline the steps taken to create and modify this project based on the original implementation.
Finally, readers can explore integrating this project as a sentence ranker option within the MRAG pipeline.

## Preparation for packaging

1. Based on the [original PreSumm](https://github.com/nlpyang/PreSumm), clone and create `setup.py` in the project root directory: PreSumm (details in code) .
2. Rename '../src' to '../presumm', i.e. ``` mv src presumm ```.
3. Create `__init__.py` inside the '../presumm' directory (details in code).

## Modifications
Please refer to the repository for detailed changes made to the codebase.

## Packaging
Install wheels, build by wheels, and packaing.
```bash
pip install wheel
cd /path/to/PreSumm
python setup.py bdist_wheel
```
There should be a  file in the `../dist/.whl` file now. Install the .whl into your environment by
```bash
pip install /path/to/xxx.whl
```


