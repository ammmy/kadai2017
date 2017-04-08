from model_simple import model
from config_simple import *
import logreg
if __name__ == "__main__":
    logreg.train(model, dim_embed=dim_embed, class_num=class_num, learning_rate=learning_rate)

