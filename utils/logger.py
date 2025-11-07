import logging
from datetime import datetime, timedelta

from utils.commn import FileUtils


def beijing(sec, what):
    beijing_time = datetime.now() + timedelta(hours=8)
    return beijing_time.timetuple()


logging.Formatter.converter = beijing


def create_logger(save_path, type='Train'):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    now = (datetime.now()).strftime("%Y%m%d%H")
    log_file = f'{save_path}/logs/{type}_log_{now}.log'
    try:
        FileUtils.make_updir(log_file)
    except:
        pass
    fileinfo = logging.FileHandler(log_file)

    controshow = logging.StreamHandler()
    controshow.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controshow)
    return logger
