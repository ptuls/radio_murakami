# -*- coding: utf-8 -*-
import logging

FORMAT = "%(levelname)s:%(asctime)s:%(message)s"


def setup_logging() -> logging.Logger:
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    return logging.getLogger()
