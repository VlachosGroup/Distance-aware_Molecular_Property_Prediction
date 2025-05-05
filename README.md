# Distance-aware Molecular Property Prediction in Nonlinear Structure-Property Space


## Abstract

Molecular property prediction with limited data in novel chemical domains remains challenging. We introduce an approach based on the hypothesis that prediction difficulty increases systematically with distance from well-characterized regions in an appropriately defined structure-property space. Our framework combines nonlinear structure-property space embedding with distance-aware domain classification and uncertainty quantification. We create a structure-property embedding connecting molecular similarity with prediction difficulty, implement distance-aware classification balancing precision and true positive rate, and provide distance-based uncertainty estimates scaled by molecular similarity. Across four ecotoxicity datasets, our local models reduced root mean squared error by 28-48% for truly in-domain molecules compared to global models, with strong correlations (r = 0.40-0.62) between distance and prediction error. In a bio-lubricant base oil property application, our approach reduced prediction error by 29% compared to a global model and outperformed transfer learning and standard machine learning approaches. This framework’s focus on relevant domains and distance-calibrated uncertainty estimates for limited, heterogeneous chemical data makes it broadly applicable across applications, such as toxicity prediction, drug discovery, and materials design.

Note: The .pkl files containing results for each application are hosted externally due to their large size.
* Link: 
* Please download each file and place it in the appropriate directory if you wish to run the corresponding analysis scripts.


