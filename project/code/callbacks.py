from keras.callbacks import Callback
import requests


class CloudCallback(Callback):
    def __init__(self, remote=False, slack_url='', stop_url='', slack_channel='', name=''):
        super(CloudCallback, self).__init__()
        self.remote = remote
        self.slack_url = slack_url
        self.slack_channel = slack_channel
        self.stop_url = stop_url
        self.name = name

    def stop_instance(self):
        if self.remote:
            self.send_update('Stopping instance, bye bye')
            requests.get(self.stop_url)

    def send_update(self, msg):
        print(msg.replace(':weight_lifter:', '🏋').replace(':tada:', '🎉'))
        if self.remote:
            payload = {'message': msg, 'channel': self.slack_channel}
            requests.post(self.slack_url, json=payload)

    # def on_train_begin(self, logs=None):
    #     self.send_update('Training {name} has just started :weight_lifter:'.format(name=self.name))

    def on_epoch_end(self, epoch, logs={}):
        # loss = logs.get('pos_loss') or logs.get('loss')
        acc = logs.get('acc')

        self.send_update('*Epoch {0} has ended*! - Accuracy: `{1}`'.format(epoch + 1, acc))
        # self.send_update('*Epoch {0} has ended*! Loss: `{1}` - Accuracy: `{2}`'.format(epoch + 1, loss, acc))

    # def on_train_end(self, logs=None):
    #     self.send_update('Training is done :tada:')
