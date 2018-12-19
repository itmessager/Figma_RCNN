apt-get install -y p7zip-full

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

MODEL_PATH=$DIR"/../models"
mkdir -p $MODEL_PATH

# Download s3fd model
wget -nc -O "$MODEL_PATH/s3fd_convert.7z" https://github.com/clcarwin/SFD_pytorch/releases/download/v0.1/s3fd_convert.7z
7z x "$MODEL_PATH/s3fd_convert.7z"
mv s3fd_convert.pth $MODEL_PATH
rm "$MODEL_PATH/s3fd_convert.7z"