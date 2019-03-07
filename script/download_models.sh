DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

MODEL_PATH=$DIR"/../model/model_epoch_190"

mkdir -p $DIR"/../model"
wget -nc -O $MODEL_PATH https://www.dropbox.com/s/qrc1c9ek737ljm5/model_epoch_190?dl=0
