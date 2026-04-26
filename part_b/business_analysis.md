B1. Problem Formulation

(a) Machine Learning Formulation

This is a Regression problem. The primary goal is to predict a continuous numerical value the target variable, which is the total number of items sold (sales volume) in a given store during a specific month. By predicting this volume for each of the five potential promotion types, the retailer can simply select the one that yields the highest number. To make these predictions accurate, the input features must capture the environment and the strategy: we need to include the "Promotion Type" being tested, "Store Characteristics" like size and location (urban vs. rural), "Market Context" such as competition density and footfall, and "Temporal Factors" like seasonality or the specific month. Regression is the ideal choice here because we are not just looking for a "yes/no" answer; we need to estimate the magnitude of success to choose the absolute best performer among several options.

(b) Target Variable Selection: Volume vs. Revenue

While total sales revenue might seem like the obvious metric for a business, using items sold is a much more reliable target for measuring promotional "pull." Revenue is often "noisy" because it is heavily influenced by the price of the products. For example, a single high-end coat sale generates more revenue than ten t-shirts, but that doesn't mean the promotion on coats was more effective at driving customer behaviour. Items sold (volume) strips away the price fluctuations and shows us exactly how much "action" a promotion created. This illustrates a vital principle in real-world ML: Target Alignment. You want to choose a variable that is most directly influenced by the feature you are testing (the promotion) and least affected by outside factors (like premium pricing). In fashion, where clearing inventory before the next season is a life-or-death priority, volume is the truest pulse of a promotion’s health.

(c) Alternative Modelling Strategy

Rather than a "one-size-fits-all" global model, a much smarter approach would be to use a Segmented or Clustered Modelling strategy. We could group the 50 stores into clusters based on their environment—such as "Urban High-Traffic," "Rural Value-Driven," and "Semi-Urban Family-Oriented" and train a specific model for each group. The reality is that a shopper in a bustling downtown metro responds to a "Loyalty Points Bonus" very differently than a shopper in a rural town might respond to a "BOGO" deal. A global model tends to "average out" these nuances, essentially becoming "decent" for everyone but "great" for no one. By segmenting the models, we allow the AI to learn the unique cultural and economic "personality" of each store type, leading to much more localised and effective promotional hits.

------------------------------------------------------------------------------------------------------------------------------------------------------------

B2. Data and EDA Strategy

(a) Data Integration and Aggregation

To build a cohesive dataset, the Transactions table would be the core and a series of left joins will be performed. I would link the Store Attributes using the Store_ID, attach the Calendar details via the Transaction_Date, and map the Promotion Details using the Promotion_ID.

The grain of the final modeling dataset would be one row per store, per month.

Since raw transactions are recorded at the individual purchase level, we need to perform several aggregations to reach this monthly grain:
Summation: Total "Items Sold" (our target variable) and total footfall per month.
Flagging/Counting: A count of "Festival Days" or "Weekend Days" in that specific month from the calendar table.
Categorization: Identifying the specific "Promotion Type" active for that store-month combination.

(b) Exploratory Data Analysis (EDA)

Promotion vs. Volume Boxplot: I will create a boxplot showing the distribution of items sold for each of the five promotion types. The target is "clear winners" if BOGO consistently has a higher median and tighter spread, it is a strong predictor. This helps in validating if the promotion categories are distinct enough to be useful features.

Seasonality Line Chart: Plotting total items sold over time, overlaid with festival flags. I will look for peaks during holidays or specific months. If certain months show massive spikes regardless of the promotion, I will need to ensure "Month" is a heavily weighted feature to prevent the model from incorrectly attributing organic seasonal growth to a specific marketing campaign.

Store Cluster Heatmap: A heatmap correlating store attributes (size, location) with sales volume. If "Urban" stores show a high correlation with "Loyalty Points" but "Rural" stores do not, this confirms the need for the segmented modeling.

Scatter Plot of Footfall vs. Items Sold: I will check for a linear relationship. If there are stores with high footfall but low sales, it suggests that the wrong promotions are being run, or there's high competition. This helps identify if "Footfall" should be a raw input or if we should engineer a "Conversion Rate" feature.

(c) Handling Promotional Imbalance

Having 80% of your data consist of "No Promotion" periods creates a significant class imbalance (or more accurately, a distribution shift). If left unaddressed, the model might become "lazy" and simply learn to predict baseline sales, failing to capture the subtle "lift" that happens when a promotion actually kicks in.

To fix this:

Resampling: I might under-sample the "No Promotion" months or over-sample the "Promotion" months to give the model more exposure to the promotional impact.

Feature Engineering (The Baseline approach): Instead of just predicting total sales, I could engineer a feature for "Expected Baseline Sales" (average sales without a promo). The model then focuses on predicting the Residual—the extra items sold specifically due to the promotion. This effectively "subtracts" the 80% baseline noise and forces the model to learn the effectiveness of the marketing intervention.

------------------------------------------------------------------------------------------------------------------------------------------------------------

B3. Model Evaluation and Deployment

(a) Validation Strategy and Metrics

When dealing with data that spans three years, we must use a Time-Series Split (or Forward Chaining) rather than a random split. A random split is inappropriate because it would likely result in "data leakage," where the model accidentally sees future information to predict the past—for example, using December 2025 trends to predict December 2024. This does not reflect the real world, where we only have historical data to predict next month. Instead, I would train on the first 30 months and test on the final 6 months to ensure the model can handle future uncertainty.

For evaluation, I would focus on two key metrics:

Mean Absolute Error (MAE): This tells us, on average, how many units we are off by (e.g., "The model is off by 50 items"). This is easy to communicate to the inventory team because it translates directly to physical stock.
Root Mean Squared Error (RMSE): This penalizes large errors more heavily than MAE. In retail, being slightly off is okay, but being wildly off is disastrous for the supply chain. A high RMSE would warn us that the model is failing significantly in specific "edge case" stores.

(b) Investigating Recommendations via Feature Importance

To explain why Store 12 gets different advice in December versus March, I would use Local Feature Importance (such as SHAP values). This allows us to see which specific "knobs" the model turned to reach its conclusion.

When communicating with the marketing team, I would not just show them raw coefficients. I would explain that in December, the "Seasonality/Festival" feature likely had the highest importance; the model learned that holiday shoppers are already in-store and responds best to "Loyalty Points" to build long-term retention. However, in March—historically a slower month—the "Local Competition" feature might become the dominant factor. The model likely sees that competitors are running clearance sales, so it recommends a "Flat Discount" to remain price-competitive and maintain footfall. Essentially, the model is shifting its strategy from "relationship building" in the peak season to "aggressive acquisition" in the off-season.

(c) End-to-End Deployment and Monitoring

Serialization: Once the model is trained and validated, I would save it as a Pickle or Joblib file. This "freezes" the model weights and architecture so it can be moved from a research environment to a live server.

The Monthly Pipeline: At the start of each month, an automated script would pull the latest data from the four tables (Transactions, Store Attributes, etc.). The script would aggregate this data into the monthly grain (one row per store), just like we did during training. This "pre-processed" data is then fed into the saved model.

Inference: The model would generate five predictions for every store—one for each promotion type. The system then outputs the promotion with the highest predicted sales volume as the final recommendation.

Monitoring and Retraining: I would implement a Dashboard to track Model Drift. Every month, I wouldd compare the model’s predicted items sold against the actual items sold. If the MAE or RMSE exceeds a pre-defined threshold (e.g., the model is consistently off by 20% more than usual), it signals that consumer behavior has shifted. At that point, the system would trigger an alert for the data science team to retrain the model on the most recent data.