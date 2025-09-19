# CrowdCast: Public Transport Crowd Predictor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random

# Why: Generate sample transport data for demonstration
def generate_sample_data(num_records=1000):
    np.random.seed(42)  # For reproducible results
    locations = ['Central Station', 'Downtown Terminal', 'University Stop', 'Shopping District', 'Sports Arena']
    
    data = []
    for i in range(num_records):
        # Random date/time in the last 30 days
        timestamp = datetime.now() - timedelta(days=random.randint(0, 30), 
                                              hours=random.randint(0, 23),
                                              minutes=random.randint(0, 59))
        
        location = random.choice(locations)
        hour = timestamp.hour
        day_of_week = timestamp.weekday()  # 0=Monday, 6=Sunday
        
        # Base crowd level with some patterns
        if 7 <= hour <= 9 or 16 <= hour <= 18:  # Rush hours
            base_crowd = random.randint(80, 150)
        else:
            base_crowd = random.randint(20, 80)
            
        # Weekend effect
        if day_of_week >= 5:  # Weekend
            base_crowd = random.randint(50, 120)
            
        # Location effect
        if location == 'Sports Arena':
            base_crowd += random.randint(0, 50)
        elif location == 'University Stop':
            if 8 <= hour <= 16:  # School hours
                base_crowd += random.randint(20, 40)
        
        # Add some randomness
        base_crowd += random.randint(-10, 10)
        base_crowd = max(10, base_crowd)  # Ensure at least 10 people
        
        data.append({
            'timestamp': timestamp,
            'hour': hour,
            'day_of_week': day_of_week,
            'location': location,
            'crowd_level': base_crowd
        })
    
    return pd.DataFrame(data)

# Why: Create and train the machine learning model
def train_model(df):
    # Prepare features (X) and target (y)
    location_dummies = pd.get_dummies(df['location'], prefix='loc')
    X = pd.concat([df[['hour', 'day_of_week']], location_dummies], axis=1)
    y = df['crowd_level']
    
    # Why: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Why: Create and train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    return model, X_test, y_test, y_pred

# Why: Evaluate model performance and create visualizations
def evaluate_and_visualize(y_test, y_pred):
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    # Why: Set up the visualization style
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Actual vs Predicted values
    plt.subplot(2, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Crowd Level')
    plt.ylabel('Predicted Crowd Level')
    plt.title('Actual vs Predicted Crowd Levels')
    
    # Plot 2: Error distribution
    plt.subplot(2, 2, 2)
    errors = y_pred - y_test
    plt.hist(errors, bins=30, edgecolor='black')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    
    # Plot 3: Feature importance
    plt.subplot(2, 2, 3)
    feature_importance = model.feature_importances_
    feature_names = X_test.columns
    indices = np.argsort(feature_importance)[::-1]
    plt.bar(range(len(feature_importance)), feature_importance[indices])
    plt.xticks(range(len(feature_importance)), [feature_names[i] for i in indices], rotation=45)
    plt.title('Feature Importance')
    
    # Plot 4: Crowd level by hour
    plt.subplot(2, 2, 4)
    hour_means = df.groupby('hour')['crowd_level'].mean()
    plt.plot(hour_means.index, hour_means.values, marker='o')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Crowd Level')
    plt.title('Average Crowd Level by Hour')
    plt.xticks(range(0, 24))
    
    plt.tight_layout()
    plt.show()

# Why: Make a sample prediction for demonstration
def make_prediction(model, hour, day_of_week, location):
    locations = ['Central Station', 'Downtown Terminal', 'University Stop', 'Shopping District', 'Sports Arena']
    location_dummies = pd.get_dummies([location], prefix='loc')
    
    # Create a row with all location columns
    all_locations = pd.DataFrame(columns=[f'loc_{loc}' for loc in locations])
    all_locations.loc[0] = 0
    
    # Set the correct location to 1
    for col in location_dummies.columns:
        if col in all_locations.columns:
            all_locations[col] = 1
    
    # Add hour and day_of_week
    all_locations['hour'] = hour
    all_locations['day_of_week'] = day_of_week
    
    # Ensure correct column order
    feature_columns = ['hour', 'day_of_week'] + [f'loc_{loc}' for loc in locations]
    all_locations = all_locations[feature_columns]
    
    prediction = model.predict(all_locations)
    return prediction[0]

# Main execution
if __name__ == "__main__":
    print("Generating sample data...")
    df = generate_sample_data(2000)  # Generate 2000 sample records
    
    print("Training Random Forest model...")
    model, X_test, y_test, y_pred = train_model(df)
    
    print("Evaluating model and creating visualizations...")
    evaluate_and_visualize(y_test, y_pred)
    
    # Example prediction
    print("\nExample prediction:")
    predicted_crowd = make_prediction(model, 17, 0, 'Central Station')  # Monday at 5 PM
    print(f"Predicted crowd level at Central Station on Monday at 5 PM: {predicted_crowd:.0f} people")
    
    # Show peak and off-peak hours
    print("\nPeak and Off-Peak Analysis:")
    hour_analysis = df.groupby('hour')['crowd_level'].mean()
    peak_hour = hour_analysis.idxmax()
    off_peak_hour = hour_analysis.idxmin()
    print(f"Peak hour: {peak_hour}:00 ({hour_analysis[peak_hour]:.0f} people on average)")
    print(f"Off-peak hour: {off_peak_hour}:00 ({hour_analysis[off_peak_hour]:.0f} people on average)")