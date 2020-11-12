import os
from datetime import datetime

from google.cloud import storage
from google.cloud.storage import Bucket
from torch.utils.tensorboard import SummaryWriter

revision = os.environ.get("REVISION") or "%s" % datetime.now()
tensorboard_dir = os.environ.get('TENSORBOARD_DIR') or '/tmp/tensorboard'
message = os.environ.get('MESSAGE')
client = storage.Client.from_service_account_json(os.path.expanduser("~/.gs/hebbian-meta-learning.json"))


def get_writers(name):
    train_writer = SummaryWriter(tensorboard_dir + '/%s/%s/train/%s' % (name, revision, message))
    test_writer = SummaryWriter(tensorboard_dir + '/%s/%s/test/%s' % (name, revision, message))
    return train_writer, test_writer


def upload_results(fname):
    bucket: Bucket = client.bucket("hebbian-meta-learning")
    bucket.blob("results/%s/%s" % (revision, fname)).upload_from_filename(fname)
