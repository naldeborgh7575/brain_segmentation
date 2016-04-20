import time
import os
import progressbar
import threading
from glob import glob
from boto.s3.connection import S3Connection
progress = progressbar.ProgressBar(widgets=[progressbar.Bar('*', '[', ']'), progressbar.Percentage(), ' '])

def files_to_s3(files, bucket_name):
    '''
    INPUT   (1) list 'files': all files to upload to s3 bucket
            (2) string 'bucket_name': name of bucket to dump into
    writes all files to s3 bucket using threads
    '''
    AWS_KEY = os.environ['AWS_ACCESS_KEY_ID']
    AWS_SECRET = os.environ['AWS_SECRET_ACCESS_KEY']


    def upload(myfile):
        conn = S3Connection(aws_access_key_id = AWS_KEY, aws_secret_access_key = AWS_SECRET)
        bucket = conn.get_bucket(bucket_name)
        key = bucket.new_key(myfile).set_contents_from_filename(myfile) # , cb=percent_cb, num_cb=1)
        return myfile

    for fname in files:
        t = threading.Thread(target=upload, args=(fname,)).start()

if __name__ == '__main__':
    files = glob('n4_PNG/**')
    progress.currval = 0
    start_time = time.time()
    for x in progress(xrange(len(files) / 100)): #avoid threading complications
        time.sleep(2)
        f = files[100 * x : (100 * x) + 100]
        files_to_s3(f, 'n4itk-slices')
        # print(str(x) + ' out of ' + str(len(files)/100))
    print('------%s seconds------' % (time.time() - start_time))
