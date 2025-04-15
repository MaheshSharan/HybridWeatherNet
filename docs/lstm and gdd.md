 Time-based patterns (LSTM)
Where: src/models/lstm_module.py
What it does:
The LSTM processes sequences of weather data (like temperature, humidity, etc.) over time for each location.
Why:
It learns how today’s weather depends on previous days (e.g., temperature trends, recurring patterns).
In code:
The LSTM receives input shaped like (batch, sequence_length, features) and outputs learned “temporal” features for each time step.
In simple terms:

“Time-based patterns are trends or cycles in weather over days or weeks, like how temperature rises and falls. LSTM learns these patterns by looking at sequences of data over time.”

2. Space-based patterns (GNN)
Where: src/models/graph_module.py
What it does:
The GNN processes data from multiple locations (weather stations) and learns how weather at one place is connected to weather at nearby places.
Why:
Weather at one station is influenced by its neighbors (e.g., a storm moving across stations).
In code:
The GNN receives a graph of stations, with edges showing which stations are “neighbors.” It updates each station’s data by mixing in information from its neighbors.
In simple terms:

“Space-based patterns are the relationships between different locations—how weather at one place affects another. GNN learns these patterns by connecting stations together and sharing information between them.”

How they work together in your model:

The LSTM learns the “when” (temporal patterns).
The GNN learns the “where” (spatial patterns).
The attention module then decides which combinations of these are most important for making the best prediction.