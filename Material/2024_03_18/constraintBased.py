import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Sample camera data
cameras_data = {
    'Camera': ['Camera A', 'Camera B', 'Camera C'],
    'Megapixels': [24, 18, 20],
    'Zoom_Range': ['20x', '30x', '15x'],
    'Sensor_Type': ['APS-C', 'Full Frame', 'Micro Four Thirds']
}
cameras_df = pd.DataFrame(cameras_data)

# Sample user preferences
user_preferences = {
    'Sensor_Type': 'Micro Four Thirds',
    'Megapixels': 18,   # Preferred megapixels
    'Zoom_Range': '15x'   # Preferred zoom range
      # Preferred sensor type
}

# Ensure user preferences have the same data type as the columns in cameras_df
for column in cameras_df.select_dtypes(include='object').columns:
    if column in user_preferences:
        user_preferences[column] = str(user_preferences[column])

# Label encode categorical attributes
label_encoders = {}
for column in cameras_df.select_dtypes(include='object').columns:
    label_encoders[column] = LabelEncoder()
    cameras_df[column] = label_encoders[column].fit_transform(cameras_df[column])
    if column in user_preferences:
        user_preferences[column] = label_encoders[column].transform([user_preferences[column]])[0]

# Calculate similarity score for each camera
def calculate_similarity(camera):
    similarity = 0
    for attribute in user_preferences.keys():
        similarity += abs(camera[attribute] - user_preferences[attribute])
    return similarity

cameras_df['Similarity'] = cameras_df.apply(calculate_similarity, axis=1)

# Sort cameras by similarity (ascending order)
recommendations_df = cameras_df.sort_values(by='Similarity')

# Recommend top N cameras
top_n = 1
top_cameras = recommendations_df.head(top_n)['Camera'].tolist()

print("Top {} recommended cameras:".format(top_n))
for camera in top_cameras:
    print(camera)
