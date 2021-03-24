import os
from datetime import datetime

from google.cloud import storage
from google.cloud.storage import Bucket
from torch.utils.tensorboard import SummaryWriter

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.expanduser("~/.gs/hebbian-meta-learning.json")
revision = os.environ.get("REVISION") or "%s" % datetime.now()
message = os.environ.get('MESSAGE')
tensorboard_dir = "gs://hebbian-meta-learning/tensorboard" if os.environ.get("REVISION") else "/tmp/tensorboard"
client = storage.Client()
flush_secs = 10


def get_writers(name):
    train_writer = SummaryWriter(tensorboard_dir + '/%s/%s/train/%s' % (name, revision, message), flush_secs=flush_secs)
    test_writer = SummaryWriter(tensorboard_dir + '/%s/%s/test/%s' % (name, revision, message), flush_secs=flush_secs)
    return train_writer, test_writer


def upload_results(fname):
    bucket: Bucket = client.bucket("hebbian-meta-learning")
    bucket.blob("results/%s/%s" % (revision, fname)).upload_from_filename(fname)
