from pathlib import Path

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import TextVectorization

from evaluate import build_classification_report, save_confusion_matrix
from model import build_siamese_bilstm_model
from predict import predict_match
from preprocessing import build_labeled_pairs, read_jsonl
