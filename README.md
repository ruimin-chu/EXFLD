# <em>EXPLD</sup></em>
### An Effective Approach for Early Fuel Leakage Detection with Enhanced Explainability

## Abstract
Leakage detection at service stations with underground storage tanks containing hazardous products, such as fuel, is a critical task. Early detection is important to halt the spread of leaks, which can pose significant economic and ecological impacts on the surrounding community. Current data-driven fuel leakage detection methods rely on statistical analysis of daily inventory log data over a month, leading to detection delays. Moreover, no existing work explores explainable methods in this domain to aid practitioners in validating detection results. While existing XAI deep learning methods for time series analysis show strong performance and offer explainability, their explanations are often difficult to interpret and not necessarily useful for validation. Therefore, we propose an EXplainable Fuel Leakage Detection approach called EXFLD, which performs online fuel leakage detection and provides intuitive explanations for detection validation. EXFLD incorporates a high-performance deep learning model for accurate online fuel leakage detection and an inherently interpretable model to generate intuitive textual explanations to assist practitioners in result validation. Through case studies, we demonstrate that EXFLD can provide intuitive and meaningful textual explanations that humans can easily understand. Meanwhile, experimental results on a fuel leakage dataset confirm that EXFLD achieves competitive performance compared to baselines in terms of accuracy.


## Scripts to Run Each Model

To train the TFT for online detection:
- `TFTTrain.py`
To run online detection for EXPLD evaluation, use the following commands:
- `python OnlineDetection.py`
The example code to generation explanations and interpretable results can be found in under Explanations folder
