from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')
        self.parser.add_argument('--save_dataset',type=str, default='img',help='save as pngs or a matfile')
        self.parser.add_argument('--dataset_name',type=str, default='',help='data on which the model will be tested')
        self.parser.add_argument('--save_folder_name',type=str, default='',help='folder for saving test results')
        self.parser.add_argument('--personalized',type=int, default=0,help='load personalized or federated model')
        self.isTrain = False
