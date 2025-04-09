# Potential Questions During Presentation

## Basic Concept Questions

1. **What exactly is weather bias?**
   - *Simple answer*: Weather bias is when there's a consistent difference between what the forecast predicts and what the actual weather turns out to be. For example, if a forecast always predicts temperatures that are cooler than reality, that's a warm bias.

2. **Why do weather forecasts have bias in the first place?**
   - *Simple answer*: Weather models are simplified versions of reality and can't capture every detail. They might not account for local factors like buildings, trees, or small hills that affect temperature. Also, the models are usually designed for large areas, not specific points.

3. **How is your project different from existing weather forecasts?**
   - *Simple answer*: We don't create new forecasts from scratch. Instead, we take existing forecasts and make them better by learning patterns in their errors and fixing them.

## Data Questions

4. **Where does your data come from?**
   - *Simple answer*: We use two main sources: Open-Meteo for weather forecasts and ISD-Lite (from NOAA) for actual observed temperatures from weather stations around the world.

5. **How much data did you need to train your model?**
   - *Simple answer*: We used about 5 years of daily weather data (2018-2023) from multiple locations, which is thousands of data points. This helps our model learn patterns across different seasons and weather conditions.

6. **How do you handle missing data?**
   - *Simple answer*: Sometimes weather stations have gaps in their readings. We use techniques like forward-fill and backward-fill, which basically means we use the closest available readings to fill in the gaps.

## Model Questions

7. **What is normalization and why is it important?**
   - *Simple answer*: Normalization makes all our different weather measurements comparable by putting them on the same scale. It's like converting everything to a 0-10 scale. Without this, the model might think temperature (which might be 0-30°C) is more important than humidity (which might be 0-100%) just because the numbers are different sizes.

8. **Can you explain what LSTM is in simple terms?**
   - *Simple answer*: LSTM stands for Long Short-Term Memory. It's a part of our model that's really good at remembering patterns over time. If it's been getting warmer for several days, the LSTM can recognize that trend and use it to make better predictions. It's like having a really good memory for weather patterns.

9. **What is a Graph Neural Network and why use it?**
   - *Simple answer*: A Graph Neural Network (GNN) understands relationships between different locations. Weather doesn't just stay in one place - it moves around! The GNN helps our model understand how weather in one place affects nearby places, like how rain might move from west to east across a region.

10. **What is the attention mechanism doing in your model?**
    - *Simple answer*: The attention mechanism helps our model focus on what's important. Not all weather features matter equally for prediction. For example, cloud cover might be super important when predicting temperature bias on sunny days, but less important on already cloudy days. Attention helps the model figure out what to focus on in different situations.

11. **How does your model handle uncertainty?**
    - *Simple answer*: We use a technique called Monte Carlo dropout, which is a fancy way of saying we run the model multiple times with slight variations. If all these runs give similar predictions, we're confident. If they give different predictions, we know there's more uncertainty.

## Results Questions

12. **How do you measure if your model is doing a good job?**
    - *Simple answer*: We use three main metrics: Mean Bias (how off we are on average), RMSE (Root Mean Square Error, which measures overall accuracy), and MAE (Mean Absolute Error, which measures the average size of our mistakes). Lower numbers are better for all of these.

13. **How much better is your model compared to regular forecasts?**
    - *Simple answer*: Regular forecasts typically have errors of 1-2°C. Our model reduces this to around 0.3-0.5°C, which is a huge improvement! That's like the difference between deciding to wear a jacket or not.

14. **Did you test your model on future data it hadn't seen before?**
    - *Simple answer*: Yes! We tested it on March 2025 data (which the model had never seen during training) and it still performed excellently, showing it can generalize to new situations.

## Technical Implementation Questions

15. **Why did you choose PyTorch over other frameworks?**
    - *Simple answer*: PyTorch gives us flexibility to build complex models and is widely used in research. It also has great tools like PyTorch Lightning that make training more organized and PyTorch Geometric for graph neural networks.

16. **How long does it take to train your model?**
    - *Simple answer*: On a good GPU, it takes a few hours. The model learns from thousands of examples, adjusting itself each time to get better at predicting temperature bias.

17. **How did you implement the web interface?**
    - *Simple answer*: We used Streamlit, which is a Python library that makes it easy to create interactive web apps. It lets users upload their own weather data and see the corrected forecasts with just a few clicks.

## Future Work Questions

18. **How could your model be improved further?**
    - *Simple answer*: We could add more weather variables like pressure or precipitation, include satellite imagery, or expand to more locations around the world. We could also adapt it to correct other weather variables like rainfall or wind speed.

19. **Could this be used in real-world applications?**
    - *Simple answer*: Absolutely! Weather services could use this to improve their forecasts, energy companies could better predict demand, and farmers could make better planting decisions. Anyone who needs accurate weather forecasts would benefit.

20. **What was the biggest challenge in building this project?**
    - *Simple answer*: Aligning the forecast data with the actual observations was tricky because they use different formats and time intervals. Also, making sure our model learned meaningful patterns rather than just memorizing the training data required careful design.
