Source: [Target Encoding Vs. One-hot Encoding with Simple Examples](https://medium.com/analytics-vidhya/target-encoding-vs-one-hot-encoding-with-simple-examples-276a7e7b3e64)

*Credit: Svideloc*

# Target Encoding

Turns this:

![1](https://miro.medium.com/max/419/1*W77md1OC9HSuAFy9b0LEIw.png)

Into this:

![2](https://miro.medium.com/max/532/1*ldULQ2FIPhkxIo56YObrfg.png)

```python
# 1. Import Libraries
import pandas as pd
from category_encoders import TargetEncoder

# 2. Target Encode & Clean DataFrame
encoder = TargetEncoder()
df['Animal Encoded'] = encoder.fit_transform(df['Animal'], df['Target'])
```

### Benefits
- Simple and quick encoding method that **doesn’t add to the dimensionality of the dataset**.
- Therefore it may be used as a good first try encoding method.

### Limitations
- Dependent on the distribution of the target → requires careful validation → prone to **overfitting**.

# One-hot Encoding

Turns into this:

![3](https://miro.medium.com/max/569/1*kz_RX5EZGXDT_gmgETtdyA.png)

```python
# 1. Import Libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import LabelEncoder

# 2. Label Encode & Clean DataFrame
le = LabelEncoder()
df['Animal Encoded'] = le.fit_transform(df.Animal)

# 3. One-hot Encode & Clean Dataframe
encoder = OneHotEncoder(categories = 'auto')
X = encoder.fit_transform(
    df['Animal Encoded'].values.reshape(-1,1)).toarray()
dfonehot = pd.DataFrame(X)
df = pd.concat([df, dfonehot], axis =1)
df.columns = ['Animal','Target','Animal Encoded',
                     'isCat','isDog','isHamster']
```

### Benefits
- Works well with nominal data.
- **Eliminates any issue of higher categorical values influencing data**.

### Limitations
- Can create very high dimensionality depending on:
  - The number of categorical features, and
  - The number of categories per feature.
  
*Note: Combining PCA with one-hot encoding can help reduce that dimensionality when running models.*
