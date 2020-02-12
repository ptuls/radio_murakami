# -*- coding: utf-8 -*-
import logging

FORMAT = "%(asctime)s\t%(levelname)s\t%(message)s"


def setup_logging() -> logging.Logger:
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    return logging.getLogger()
