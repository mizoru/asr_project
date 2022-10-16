# ASR project for DeepSpeech training

## Installation guide

You first run this in your shell:
```shell
git clone https://github.com/mizoru/asr_project.git
cd asr_project
pip install -r requirements.txt
pip install https://github.com/kpu/kenlm/archive/master.zip
```

Copy [this folder](https://drive.google.com/drive/folders/1L2oh-kxpbHP15CCRYbea9m9o6kEPEDam) so that you have `asr_project/default_test_model/default_checkpoint.pth` as the checkpoint path.

You can now run this line to get predictions on files in `test_data`:
```shell
python test.py \
   -c default_test_config.json \
   -r default_test_model/checkpoint.pth \
   -t test_data \
   -o test_result.json \
   -b 5
```
or this line to obtain the metrics on the test part of LibriSpeech:
```shell
python test_metrics.py \
   -c default_test_config.json \
   -r default_test_model/checkpoint.pth  \
   -o test_metrics.json
```

## Credits

This repository is based on a modified, filled-in fork of [asr_project_template](https://github.com/WrathOfGrapes/asr_project_template) repository.

## TODO

* Add a batch sampler
