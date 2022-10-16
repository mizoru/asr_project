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

You can now run this line to get predictions on files in `test_data`
```shell
python test.py \
   -c default_test_config.json \
   -r default_test_model/checkpoint.pth \
   -t test_data \
   -o test_result.json \
   -b 5
```

## Before submitting

0) Make sure your projects run on a new machine after complemeting the installation guide or by 
   running it in docker container.
1) Search project for `# TODO: your code here` and implement missing functionality
2) Make sure all tests work without errors
   ```shell
   python -m unittest discover hw_asr/tests
   ```
3) Make sure `test.py` works fine and works as expected. You should create files `default_test_config.json` and your
   installation guide should download your model checpoint and configs in `default_test_model/checkpoint.pth`
   and `default_test_model/config.json`.
   ```shell
   python test.py \
      -c default_test_config.json \
      -r default_test_model/checkpoint.pth \
      -t test_data \
      -o test_result.json
   ```
4) Use `train.py` for training

## Credits

This repository is based on a modified, filled-in fork
of [asr_project_template](https://github.com/WrathOfGrapes/asr_project_template) repository.

## Docker

You can use this project with docker. Quick start:

```bash 
docker build -t my_hw_asr_image . 
docker run \
   --gpus '"device=0"' \
   -it --rm \
   -v /path/to/local/storage/dir:/repos/asr_project_template/data/datasets \
   -e WANDB_API_KEY=<your_wandb_api_key> \
	my_hw_asr_image python -m unittest 
```


## TODO

* Add a batch sampler
