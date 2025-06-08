## 1.9 Research Hypothesis

This research is guided by several key hypotheses that will be tested through experimentation and analysis:

1. **H1**: A hybrid approach combining supervised classification and unsupervised anomaly detection techniques will achieve significantly higher fraud detection performance (as measured by F1-score and AUC) compared to either approach used in isolation.

2. **H2**: Deep learning models implemented within a distributed Spark framework will demonstrate superior scalability characteristics compared to traditional fraud detection systems when processing large-scale banking transaction data, maintaining consistent performance as data volumes increase.

3. **H3**: Advanced sampling techniques coupled with specialized loss functions will significantly improve model performance on the minority fraud class compared to standard approaches, increasing recall without proportional degradation in precision.

4. **H4**: Feature engineering that incorporates temporal behavioral patterns and aggregated customer profiles will produce more discriminative features for fraud detection compared to approaches that consider only individual transaction attributes.

5. **H5**: Models with adaptive capabilities will maintain performance levels over time in the presence of evolving transaction patterns, showing significantly less degradation compared to static models.

These hypotheses establish testable propositions that align with the research questions and objectives. The methodology and experimental design will be structured to systematically evaluate each hypothesis, providing empirical evidence to support or refute these assertions.

## 1.10 Structure of the Thesis

This thesis is organized into five chapters that progressively develop the research from problem formulation through implementation and evaluation to conclusions. The structure is as follows:

**Chapter 1: Introduction**
This chapter introduces the research context, problem statement, and objectives, establishing the foundation for the investigation into anomaly-based fraud detection using deep learning and Apache Spark.

**Chapter 2: Literature Review**
This chapter critically examines existing research and practices in fraud detection, with particular focus on:
- Traditional vs. anomaly-based approaches
- Applications of big data technologies in financial fraud detection
- Deep learning techniques for anomaly detection
- Spark-based architectures for large-scale data processing
- Current trends and research gaps

**Chapter 3: Methodology**
This chapter details the research design and methods, including:
- Data collection and preprocessing approaches
- Feature engineering techniques
- The proposed anomaly detection framework
- Implementation of Spark-based data processing pipelines
- Deep learning model architectures
- Evaluation methodologies and metrics
- Ethical considerations in fraud detection research

**Chapter 4: Results and Discussion**
This chapter presents the experimental results and their analysis, including:
- Performance of the Spark-based data processing pipeline
- Deep learning model evaluation results
- Comparative analysis with baseline methods
- Discussion of findings in relation to the research hypotheses
- Implications for the banking sector
- Limitations of the current approach

**Chapter 5: Conclusion and Future Work**
This chapter summarizes the key findings and contributions of the research, offers recommendations for implementation in banking environments, acknowledges limitations, and suggests directions for future research.

**Appendices**
The thesis includes appendices containing:
- Detailed technical specifications
- Additional experimental results
- Code samples and implementation details
- Supplementary analyses

This structure provides a logical progression from problem definition through solution development to evaluation and conclusions, ensuring a comprehensive treatment of the research topic.
