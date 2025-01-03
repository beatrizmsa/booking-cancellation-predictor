# Hotel Booking Cancellation Prediction Project

## Project Overview
This project aims to develop predictive models capable of determining whether hotel clients will cancel their bookings. By analyzing historical booking data, we seek to uncover key patterns and relationships that can help hotels mitigate the financial impact of cancellations.

**Objective:** 
To build and evaluate machine learning models that predict booking cancellations based on various factors such as booking lead time, customer type, market segment, and more. The project follows the CRISP-DM methodology, which includes data exploration, pre-processing, model development, and evaluation.

## Dataset:

The dataset contains various features:
- **hotel**: Hotel (Resort Hotel or City Hotel). 
- **is_canceled**: Value indicating if the booking was canceled (1) or not (0). 
- **lead_time**: Number of days that passed between the booking date and the arrival date. 
- **arrival_date_year**:Year of arrival date. 
- **arrival_date_month**: Month of arrival date. 
- **arrival_date_week_number**: Week number of year for arrival date. 
- **arrival_date_day_of_month**: Day of arrival date. 
- **stays_in_weekend_nights**: Number of weekend nights the guest booked to stay at the hotel. 
- **stays_in_week_nights**: Number of week nights the guest booked to stay at the hotel. 
- **adults**: Number of adults. 
- **children**: Number of children. 
- **babies**:Number of babies. 
- **meal**: Type of meal booked. 
- **country**: Country of origin. 
- **market_segment**: Market segment designation. 
- **distribution_channel**: Booking distribution channel. 
- **is_repeated_guest**: Value indicating if the booking name was from a repeated guest (1) or not (0). 
- **previous_cancellations**: Number of previous bookings that were cancelled by the customer prior to the current booking. 
- **previous_bookings_not_canceled**: Number of previous bookings not cancelled by the customer prior to the current booking. 
- **reserved_room_type**: Code of room type reserved. 
- **assigned_room_type**: Code for the type of room assigned to the booking. 
- **booking_changes**: Number of changes made to the booking.
- **deposit_type**: Indication on if the customer made a deposit to guarantee the booking.
- **agent**: ID of the travel agency that made the booking. 
- **company**: ID of the company that made the booking.
- **days_in_waiting_list**: Number of days the booking was in the waiting list before it was confirmed to the customer. 
- **customer_type**: Type of booking.
- **adr**: Average Daily Rate as defined by dividing the sum of all lodging transactions by the total number of staying nights. 
- **required_car_parking_spaces**: Number of car parking spaces required by the customer. 
- **total_of_special_requests**: Number of special requests made by the customer. 
- **reservation_status**: Reservation last status.
- **reservation_status_date**:Date at which the last status was set.
- **name**: Name of the person that made the reservation.
- **email**: Email of the person that made the reservation.
- **phone-number**: Phone-number of the person that made the reservation.
- **credit_card**: Credit-card number of the person that made the reservation.


## Technologies Used
- Python 3.11
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## Project Structure
The project is organized as follows:
- **hotel_booking.csv**: Contains the data used for analysis.
- **notebook.ipynb**: Jupyter notebook for this project.
- **requirements.txt**: List of required dependencies for the project.
- **README.md**: This file, providing an overview of the project.

## Installation and Setup
To run this project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/beatrizmsa/booking-cancellation-predictor.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook.


## Results and Conclusion

Through several iterations, we found that our second approach improved model performance significantly.

Undersampling led to poorer results, likely due to the data imbalance causing loss of valuable information, which hindered the model’s ability to generalize accurately. This confirmed the need for balanced data to achieve reliable results.

We chose a holdout approach, as the dataset was large, and used stratified sampling in train-test splits to better handle class imbalance. During hyperparameter tuning, cross-validation in grid search helped us ensure that our results were generalizable.

We explored multiple models and combinations, fine-tuning each one. In the Weighted Average Ensemble (WAE) model, we manually adjusted class weights to find the optimal configuration. Feature engineering helped reduce noise, allowing us to keep the most useful features and add new, relevant ones.

Evaluating model performance with a variety of metrics and visualizations helped us measure the impact of each adjustment. These evaluations guided our approach, confirming which changes enhanced accuracy and effectiveness.

In the end, the Random Forest model stood out as the best performer, showing strong potential for accurately predicting hotel cancellations. This model could provide valuable insights for managing and reducing cancellations in hotel operations.

## Authors
- [Alexandre Marques - 1240435](https://github.com/AlexandreMarques27)
- [Beatriz Sá - 1240440](https://github.com/beatrizmsa)
