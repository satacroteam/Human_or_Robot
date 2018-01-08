from extract_features import Extract
from model import Model

extract = Extract()
extract.extract()
extract.build_answer()

model = Model()
model.train()
model.test()