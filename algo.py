import pickle

# Load the uploaded model to inspect its type and details
#model_file_path = 'D:\Desktop_Data\Fetal_health\ai ml project\fetal_health.pkl'

with open('improved_fetal_health.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Check the type and details of the model
model_type = type(loaded_model)
model_details = str(loaded_model)

model_type, model_details
