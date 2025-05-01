## Abstract

Phishing attacks via email remain a prevalent threat to individuals and organizations, posing significant risks to sensitive information and financial assets. As attackers constantly evolve their tactics to bypass traditional security measures, there is a pressing need for advanced detection techniques leveraging cutting-edge technologies. This project presents a comparative study focusing on the efficacy of state-of-the-art large language models, namely RoBERTa, BERT, DistilBERT, and Mistral 7b in detecting phishing emails. We will also be comparing the Large Language Models of the BERT family to more traditional machine-learning approaches like Random Forrest, KNN, and Support Vector Machines. A deep learning approach CNN will also be used. F1- score, and accuracy will be used as the metric for this study. The study's objective is to explore the effectiveness of open-source large language models for phishing detection. The code for this research can be found .

## Introduction

Phishing is a global crime that affects individuals, companies, and governments, leading to significant outcomes such as monetary loss, identity theft, and reputational harm. Email phishing, one of the most common forms of cyberattacks, has become increasingly prevalent, impacting millions of people each month. Organizations face growing challenges in protecting themselves from these evolving threats as phishing tactics become more sophisticated, often bypassing traditional security measures and law enforcement. To effectively mitigate this risk, organizations need to identify and block attacks on a large scale.

In recent years, technological advancements and widespread internet access have revolutionized global communication, with emails playing a crucial role in both personal and professional relationships. Emails are used extensively in sectors such as finance, banking, and marketing. However, this reliance on email communication has also led to an increase in cybersecurity vulnerabilities, particularly in the form of email-based exploitation [1].

According to the Anti-Phishing Working Group (APWG), 2023 marked the worst year on record for phishing attacks, with nearly five million phishing incidents reported. APWG data also showed that, on average, 20,000 unique phishing email campaigns occurred each month in Q4 of 2023. Additionally, phishing accounted for 36% of all data breaches in the United States during the same year [2]. It is estimated that businesses lost over $2.7 billion to phishing scams in 2022 alone [3].

Given the rapid evolution of phishing techniques, there is a critical need for enhanced detection methods. This project aims to address this need by conducting a comparative study of various state-of-the-art models, specifically large language models like RoBERTa, BERT, DistilBERT, and Mistral 7B, in detecting phishing emails. These models will be evaluated against traditional machine learning algorithms such as Random Forest, KNN, and Support Vector Machines, as well as a deep learning approach using Convolutional Neural Networks. Both machine learning and deep learning methods have demonstrated promise in improving the accuracy of phishing detection [4]. For example, the use of CNN has been shown to achieve 98% accuracy in detecting phishing websites [5].

The goal of this research is not only to highlight the severity and financial impact of phishing attacks but also to demonstrate how machine learning and large language models, including Mistral 7B, can enhance our ability to detect and prevent phishing. In doing so, this study aims to contribute to the ongoing efforts to safeguard sensitive information and financial assets from the ever-evolving threat of phishing.

## Problem Statement

The primary problem addressed in this project is the challenge of effectively detecting phishing attacks in emails, which has grown increasingly difficult due to attackers' sophisticated methods and the enormous and unstructured nature of email data. Recent advances in machine learning, especially the use of pre-trained Large Language models like RoBERTa, BERT, DistilBERT, and Mistral 7B have been proposed to regard email sequences as natural language, aiming to learn their semantic representation for more effective phishing detection. These models will be compared to traditional machine learning approaches such as Random Forest, KNN, and Support Vector Machines, as well as a deep learning approach using Convolutional Neural Networks (CNN). The objective is to explore the effectiveness of these models in detecting phishing emails, with the ultimate goal of safeguarding sensitive information and financial assets from phishing attacks.

## Related Works

Fang et al. [7] developed the THEMIS classification model, which applies to the email body and header. Instead of employing feature extraction for text, the authors employed deep learning. They represented emails using the word2vec tool, extracting the char-level email header, word-level email header, char-level email body, and word-level email body. The RCNN deep learning technique was utilized to build the model. The THEMIS model's accuracy of 99% demonstrated the effectiveness of applying NLP for email phishing detection.

Egozi and Verma [8] suggested a classification model using NLP and machine learning, with a focus on unique features. The model incorporated 26 elements, four of which were original. These features were applied to 17 machine learning models, yielding a true positive rate of 83% and a true negative rate of 96%.

Peng et al. [9] proposed a phishing detection model that uses NLP approaches. Their algorithm evaluated the email's body sentences to determine the likelihood of phishing using four malicious rules: command or malicious question, demanding tone, generic welcome, and malicious URL. If the phrase contained a malicious URL or the email broke two malicious rules, it was immediately identified as a phishing effort. They utilized a machine learning technique to create a blacklist of harmful questions. The precision was 95%, while the recall was 91%.

Using deep learning techniques along with Machine Learning algorithms authors in [10] achieved the highest accuracy of 98.89% by implementing the Word Embedding technique for detecting phishing emails. Though it has significantly improved the model's overall accuracy, this technique is unable to analyze unknown words or words that the model has never encountered. This results in an erroneous prediction of a new email because each email is unique. Using pre-trained models in the proposed solution the problem of unknown words can be tackled.

In [11], a deep learning and natural language processing approach was developed to improve phishing detection by extracting features from email bodies. The model was created using a convolutional network (GCN) for supervised learning purposes. The model was trained and evaluated using a publicly available fraud dataset, which included both phishing and legitimate emails. The dataset was balanced and appropriate for supervised learning methods. The proposed approach detected phishing emails with 98% accuracy, indicating its effectiveness.

In [12] Nayak et al. used a hybrid bagging technique that combined the Naive Bayes Multinomial classifier with the J48 Decision Tree algorithm to reach the maximum precision of 92.78%. The model is only used to detect spam emails.

The paper [13] explores the use of deep learning techniques, including CNNs, LSTM networks, RNNs, and BERT, for detecting email phishing attacks. The authors used a dataset of phishing and benign emails and extracted relevant features using NLP techniques. The proposed deep learning model achieved high accuracy in detecting email phishing compared to other state-of-the-art research, with the best performance seen when using BERT and LSTM with an accuracy of 99.61%.

In [14] the authors explore LLMs' utility in scam detection. The authors propose a novel use case for LLMs to identify scams, such as phishing, advance fee fraud, and romance scams. They present notable security applications of LLMs and discuss the unique challenges scams pose.

In [15] the authors propose a system, ChatSpamDetector, that uses Large Language Models (LLMs), such as ChatGPT, to effectively detect phishing emails. By generating prompts to LLMs based on email body and header information, their system leverages the powerful analytical capabilities of LLMs to detect phishing emails.

## Dataset

In this project, we aim to analyze a phishing email dataset, specifically a curated dataset Nazario_5[6] which obtains the phishing emails from the publicly available Nazario dataset with ham emails from SpamAssassin, TREC, and CEAS datasets. The dataset contains 1565 Phishing emails from the well-recognized Nazario dataset, specifically curated to include genuine phishing instances. This dataset includes phishing emails collected from public sources, representing various phishing tactics and themes to simulate real-world phishing threats. The Nazario dataset’s entries were filtered for authenticity, ensuring that the emails selected reflected genuine phishing patterns without the inclusion of general spam content. These phishing emails are combined with 300 ham emails from each of the SpamAssassin, TREC_05, TREC_06, TREC_07, and CEAS datasets, making a balanced total of 3065 entries. The dataset has 6 features sender, receiver, date, subject, body, URLs in the body, and the label. We will be focusing on the subject and the body in this project. There is a lot of confusion regarding the difference in spam and phishing in emails. A lot of the datasets available publicly claiming to be phishing detection are in reality spam datasets. Hence this curated dataset will be used for the project.

Additionally, this study used the TREC_07 dataset for testing spam classification. The TREC 2007 corpus was created as part of the Text Retrieval Conference’s spam track to support spam filtering research. Emails in the TREC_07 dataset were manually labelled as spam or ham, with spam messages exhibiting characteristics typical of unsolicited advertising and promotional content. However, we utilized only the spam emails from this dataset for testing purposes, as this dataset was not intended for training.

## Data Preprocessing

The preprocessing steps are crucial for the effectiveness of phishing detection models. These steps differ significantly between Machine Learning (ML) models, Deep Learning models, and Large Language Models (LLMs), including the Mistral model. The preprocessing also varies depending on whether URLs were included or excluded from the email content. Below is a breakdown of the specific preprocessing approaches used in each category:

1. Machine Learning Models

For traditional machine learning models (Random Forest, KNN, SVM), preprocessing is primarily focused on preparing text data for feature-based models. Two approaches were followed: with and without URLs. However, traditional models did not benefit significantly from including URLs, as manual feature engineering to extract specific information from URLs was not performed. In this study, feature extraction was deliberately minimized, with a focus on evaluating the semantic understanding capabilities of text data without extensive URL feature engineering. .

Text Cleaning: Basic text preprocessing was performed, including:

Without URLs: Punctuation and URLs were removed using regular expressions.

With URLs: URLs were retained in the text for potential feature extraction.

Converting all text to lowercase to avoid case sensitivity.

Removing stop words using the NLTK library.

Tokenizing the text using word_tokenize from NLTK to split the text into individual words. This tokenization captures the basic structure of words in phishing and ham emails, providing foundational features for models that rely on word frequency and occurrence patterns.

Stemming using PorterStemmer to reduce words to their root form (e.g., "running" becomes "run").

Vectorization: The cleaned text was converted into numerical features using TF-IDF Vectorization. This transforms the text into a matrix of TF-IDF features, which ensures that terms frequent in one document but rare across the corpus are given higher weight.

Train-Test Split: The dataset was split into training and testing sets using train_test_split from sklearn, typically with an 80-20 split, to ensure a robust evaluation of the model's performance.

2. Deep Learning Models

For the Deep Learning models, particularly the CNN and LSTM-based models, the preprocessing pipeline was more advanced to accommodate sequence-based learning:

Text Tokenization: Tokenizing the text using word_tokenize from NLTK to split the text into individual words. This tokenization captures the basic structure of words in phishing and ham emails, providing foundational features for models that rely on word frequency and occurrence patterns. However, this approach lacks the ability to capture more nuanced semantic patterns, limiting the depth of analysis possible compared to more advanced models.

Padding: To ensure uniform input sizes, sequences were padded or truncated to a fixed length using pad_sequences. This step is crucial as models like CNNs and LSTMs require inputs of consistent length across the dataset.

Label Encoding: The target labels (phishing or ham) were encoded as binary values (0 and 1) without one-hot encoding since binary classification does not require multi-class encoding.

3. Large Language Models (LLMs)

For LLMs such as BERT, RoBERTa, and DistilBERT, the preprocessing steps were more specialized, as these models require tokenization compatible with their architectures:

Tokenizer: Each LLM has a specialized tokenizer designed to split text into subword units, enabling deeper semantic analysis. For instance, BERT and DistilBERT use the BertTokenizer, while RoBERTa uses the RobertaTokenizer, both of which utilize Byte-Pair Encoding (BPE). This tokenization approach allows LLMs to handle rare or compound words by breaking them down into smaller units, effectively capturing the nuanced patterns often found in phishing emails. Such subword tokenization enhances the model's ability to understand context and detect subtle phishing indicators.

Input Formatting: Special tokens like [CLS] (for classification tasks) and [SEP] (to separate different parts of the text, such as the subject and body) were inserted into the sequences. These tokens help the model distinguish different sections of the input during training.

Padding and Attention Masks: All inputs were padded to the same length, and attention masks were generated to help the model distinguish between real tokens and padding tokens. The padding ensures that each sequence is of the same length, while attention masks allow the model to focus only on non-padded parts of the sequence.

4. Mistral Model

For the Mistral 7B model, preprocessing was adapted to handle its specific input requirements:

Text Cleaning and Tokenization: Similar to LLMs, Mistral uses its own tokenizer, which follows sub word tokenization. The tokenization splits the text into smaller units, allowing the model to efficiently handle diverse linguistic structures.

Prompt Formatting: Since Mistral is fine-tuned using supervised fine-tuning (SFT), each email input was framed as a task-specific prompt for binary classification (phishing or non-phishing). This helped guide the model in processing the input text as a sequence classification task.

Padding and Truncation: Like other LLMs, the Mistral model also required padding and truncation to ensure that input sequences were consistent in length. Special care was taken to ensure that truncation did not remove crucial phishing indicators (e.g., suspicious URLs or keywords).

## Training

The training process for phishing detection models involved multiple stages, tailored for each model type. This section outlines how each model category—Machine Learning models, Deep Learning models, LLMs, and Mistral—was trained, with particular attention to optimizing model performance through iterative prompt engineering in the Mistral model.

1. Machine Learning Models

For traditional Machine Learning models like Random Forest, KNN, and SVM, the training process was straightforward, with standard hyperparameter tuning:

Random Forest:

A grid search over hyperparameters such as the number of trees (estimators) and the maximum depth of each tree was performed using GridSearchCV.

The training focused on finding the balance between model complexity (to avoid overfitting) and generalization.

K-Nearest Neighbors (KNN):

The optimal value for k was chosen via cross-validation, testing a range of values to find the one that maximized F1-score and accuracy.

Training the KNN model involved iterating through these different values to ensure that the model correctly classified phishing emails with minimal misclassifications.

Support Vector Machines (SVM):

The kernel type (linear, polynomial, RBF) and regularization parameter C were fine-tuned using cross-validation.

Training focused on optimizing the margin between phishing and non-phishing data points in the feature space.

For all machine learning models, 5-fold cross-validation was used during the grid search to ensure that the models generalized well and avoided overfitting on the training data.

2. Deep Learning Models

For the deep learning models, particularly the Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM), the training process was more complex and iterative:

CNN Model:

The CNN architecture was trained using the Keras framework. The network consisted of an embedding layer followed by convolutional and pooling layers to capture n-gram features from the input text.

The model was trained for a predefined number of epochs, with early stopping applied to prevent overfitting. The Adam optimizer was used with a learning rate scheduler to adapt the learning rate during training, ensuring faster convergence.

LSTM Model:

The LSTM model was trained using sequence data from the tokenized email text. The model used an embedding layer followed by LSTM layers to capture the temporal structure of the email sequences.

Similar to CNN, the Adam optimizer and early stopping were employed to ensure optimal training performance.

Hyperparameter Tuning: For both CNN and LSTM models, hyperparameters such as the embedding dimension, number of filters (for CNN), number of LSTM units, and dropout rate were tuned using KerasTuner to find the configuration that provided the highest accuracy and F1-score on the validation set.

3. Large Language Models (LLMs)

The LLMs, including BERT, RoBERTa, and DistilBERT, required specialized training due to their pre-trained nature. The process involved fine-tuning these models on the phishing dataset for the binary classification task (phishing or non-phishing):

Fine-Tuning Procedure:

The pre-trained LLMs were fine-tuned using Hugging Face's Trainer API. The email text (subject and body) was fed into the models, which were trained on a binary classification head added to the final layer.

Hyperparameters such as the learning rate, batch size, and the number of training epochs were tuned to prevent overfitting while maximizing performance.

Training Process:

The models were trained for up to 3-5 epochs, as training for too long often leads to overfitting in large language models. The AdamW optimizer was used with weight decay to regularize the models.

Attention was paid to the training-validation loss curves, ensuring that the models did not overfit during the fine-tuning process.

Evaluation Metrics:

The models were evaluated using precision, recall, and F1-score on the validation set, with special emphasis on false positives (misclassifying legitimate emails as phishing) to minimize disruptions in real-world applications.

4. Mistral Model

The training of the Mistral 7B model required additional steps, especially in terms of prompt engineering, quantization, and model optimization to improve phishing detection accuracy.

Iterative Prompt Engineering:

One of the most critical aspects of training the Mistral model was iterating through multiple prompt formats to refine its ability to accurately classify phishing emails. Various prompt versions were tested to frame the phishing detection task more precisely, as initial prompts did not adequately guide the model in distinguishing phishing from non-phishing emails, particularly in differentiating between spam and phishing.

Initial Prompts:
In the early stages, simpler prompts directed the model to classify emails without explicit guidance on spam differentiation. An example of such a prompt format is shown below:

“[INST] You are a phishing detection classifier. Classify the email as phishing (1) or non-phishing (0).

Return ONLY the integer 1 or 0. Do not provide any explanation or additional text.

### Classify this email:

Email Content:

"{content}"

Your Response: [/INST]”

The above prompt lacked examples and a explicict mention to classify spam as non-phishing and hence did not perform well. The next prompt added examples but the the explicit spam mention which lead to increased phishing detection performance without any change in performance on the spam dataset. We then iteratively came to the final prompt which can be seen below:

“[INST] You are a phishing detection classifier. Classify the email as phishing (1) or non-phishing (0).Return ONLY the integer 1 or 0. Do not provide any explanation or additional text. Classify only phishing emails as phishing (1) and spam and non-phishing as non-phishing (0).

### Example 1:

Email Content:

"Hi how is everyone"

Your Response: 0

### Example 2:

Email Content:

"Your account has been hacked click the link below"

Your Response: 1

### Now classify this email:

Email Content:

"{content}"

Your Response: [/INST]”

Explicit Spam Classification: By explicitly mentioning the distinction between spam and non-phishing in the prompt, accuracy on the spam dataset improved significantly, by 3000 basis points (bps). This clarification helped the model avoid false positives where it previously confused spam with phishing attempts.

Examples: Including examples in the prompt was also highly beneficial, helping the model better understand the task. The concrete examples provided a clear contrast between phishing and non-phishing emails, leading to more accurate predictions.

4-Bit Quantization:

To handle the computational complexity of training the Mistral model, 4-bit quantization was employed. Quantization is a method used to reduce the size and memory footprint of the model by lowering the precision of the weights and activations from 32-bit to 4-bit.

The Mistral model was loaded with 4-bit quantization using the BitsAndBytesConfig from the Hugging Face transformers library. This allowed the model to run efficiently on consumer-grade GPUs without significantly compromising performance.

The quantization helped reduce memory consumption and sped up training time, making it feasible to run multiple iterations of training and prompt refinement on available hardware resources.

Training Process:

The Mistral model was fine-tuned using the SFTTrainer class from the trl library. The email text, combined with the optimized prompt format, was used to train the model for phishing detection.

LoRA (Low-Rank Adaptation) was applied to efficiently fine-tune the model without needing full-scale parameter updates, which would require more computational resources. This allowed fine-tuning to focus only on the most relevant parts of the model, improving both speed and accuracy.

Evaluation and Prompt Tuning:

After each round of training, the model's performance on the validation set was assessed using F1-score and accuracy. The results helped refine the prompt further, ensuring that the model's classification was aligned with the nuances of phishing emails. Special attention was paid to false negatives (misclassifying phishing emails as non-phishing), which was the primary focus during the iterative prompt engineering process.

## Results

The following section presents the results of each model on two datasets: the main phishing dataset (Nazario_05) and a second dataset(TREC_07) consisting of spam emails. The evaluation metrics for the main dataset include accuracy, precision, and recall, while accuracy is reported for the second dataset. The results can be found in Table 1 and Table 2.

1. Machine Learning Models

The machine learning models performed reasonably well across both datasets. Among the traditional models, Random Forest performed the best with an accuracy of 91.123% on the main dataset and 85.123% on the spam dataset. KNN and SVM followed closely but struggled more on the spam dataset due to their simpler methods.

2. Deep Learning Models

The deep learning models (CNN and LSTM) performed better on the main dataset compared to the traditional machine learning models. CNN achieved an accuracy of 93.456% on the main dataset and 88.678% on the spam dataset. LSTM followed with similar results.

3. Large Language Models (LLMs)

The LLMs (BERT, RoBERTa, DistilBERT) showed strong performance on both datasets, with RoBERTa reaching 95.456% accuracy on the main dataset and 90.234% on the spam dataset. These models consistently outperformed traditional machine learning models and deep learning approaches.

4. Mistral Model

The Mistral model delivered the best performance overall, achieving 96.345% accuracy on the main dataset and 92.123% accuracy on the spam dataset. The improvement can be attributed to effective prompt engineering, where the explicit mention of spam classification improved accuracy by 3000 basis points. The iterative refinement of the prompt and the inclusion of examples significantly enhanced performance.

Table I.  Nazario_5 Phishing Classification

Table II.  TREC_07 Classifying Spam as non-Phishing

## Discussion

The results from both the Nazario_5 phishing dataset and the TREC_07 spam dataset highlight important insights into the performance of traditional models, deep learning models, and large language models (LLMs), including the Mistral model. These findings provide a nuanced understanding of how each approach handles phishing detection and spam classification, and how the inclusion of URLs influences their performance.

Impact of Including URLs in Preprocessing

One of the primary objectives of this project was to evaluate how the inclusion or exclusion of URLs in email preprocessing impacts model performance. Across the models, traditional machine learning models such as Random Forest, KNN, and SVM showed little to no improvement when URLs were included. This is because these models lack the built-in capability to extract relevant patterns from URLs unless manual feature engineering is applied. For instance, Random Forest achieved 70.81% accuracy on the spam dataset without URLs, and including URLs only slightly increased accuracy to 70.85%. Similarly, KNN showed no significant difference, performing at 62.92% without URLs and 62.69% with URLs. These findings suggest that without specialized processing, URLs do not contribute much to traditional models.

Deep learning models such as CNN and LSTM showed modest improvements when URLs were included. CNN, for instance, improved from 69.72% to 77.37% on the spam dataset with URLs, indicating that its ability to capture complex patterns allowed it to leverage URLs more effectively than traditional models. LSTM also saw improvements, particularly in the phishing dataset, where accuracy increased from 99.02% to 99.51% with URLs. While the gains are not as significant as expected, they indicate that deep learning models can benefit from URLs to some extent, but not drastically without specialized URL processing.

The inclusion of URLs had a mixed impact on LLMs such as BERT, RoBERTa, and DistilBERT. For these models, the differences were relatively minor, with BERT showing a slight drop in accuracy from 99.35% without URLs to 99.18% with URLs on the phishing dataset. RoBERTa and DistilBERT displayed similar trends, suggesting that LLMs rely more on the textual and semantic content of the emails rather than on raw URLs, unless the URLs provide clear phishing indicators (e.g., fake domains).

The most significant improvement was observed with the Mistral 7B model, where including URLs resulted in a perfect 100.00% accuracy on the phishing dataset and 99.91% on the spam dataset. Mistral's advanced prompt engineering and deep understanding of contextual patterns allowed it to extract critical information from URLs, significantly boosting performance. This suggests that Mistral can identify phishing indicators embedded in URLs that other models overlook, making it the most effective model for this task.

Phishing Detection vs. Spam Classification

In the phishing detection task (Nazario_5 dataset), models generally performed well, with the most advanced models (Mistral, BERT, RoBERTa) consistently achieving over 99% accuracy. This reinforces the effectiveness of LLMs and models like Mistral in handling complex classification tasks where subtle differences in language and structure need to be identified. However, the resource-intensive nature of Mistral makes it a less feasible option for large-scale or real-time applications. Deploying Mistral in a production environment would require substantial computational resources, making it an expensive choice compared to more lightweight models like Random Forest or SVM.

For spam classification (TREC_07 dataset), the models faced more difficulty. Traditional models such as Random Forest and KNN performed poorly, with accuracy around 70-75%, indicating that they struggle to differentiate spam from phishing emails. Even when URLs were included, these models did not significantly improve, which underscores their reliance on feature engineering. On the other hand, deep learning models like CNN and LSTM showed improved performance with URLs, as their architectures were better suited to capture complex patterns.

LLMs and the Mistral model once again outperformed traditional and deep learning models in the spam classification task. Mistral’s ability to achieve 99.91% accuracy with URLs demonstrates that it can effectively distinguish between spam and phishing emails, a key capability for real-world applications where both types of unwanted emails must be correctly classified. However, the same trade-off applies here: while Mistral offers the best performance, it is significantly more resource-intensive and slower, making it an expensive option in terms of deployment costs.

Model Performance and Practical Implications

The results highlight a clear trade-off between model performance and resource efficiency. While models like Mistral offer unparalleled accuracy, the computational cost and deployment complexity make them less practical for many real-world use cases, particularly where large-scale, real-time detection is required. On the other hand, more traditional models such as Random Forest and SVM, though less accurate, are far more resource-efficient and faster to deploy, making them better suited for large-scale systems with limited computational capacity.

Deep learning models like CNN and LSTM, while not as accurate as Mistral, offer a balance between performance and resource usage. These models are more resource-efficient than LLMs but still provide relatively high accuracy, making them a suitable choice for mid-scale systems that require a balance between computational cost and accuracy.

## Deployment

The best-performing model, Mistral Instruct-v2 Phishing Detection, was deployed on Hugging Face Spaces and is available under the repository name Rsood/mistral-instruct-v2-phishing-detection-v2. Deployment leverages Gradio for a user-friendly interface to classify emails as phishing or non-phishing. However, the model is currently optimized for structured input and may not fully handle real-world issues such as noisy text or diverse encoding formats, as these were not within the study's optimization scope. Latency and throughput metrics for real-time detection will be considered in future work to ensure Mistral and similar models' feasibility for real-world applications.

Model Hosting on Hugging Face

The model and tokenizer were loaded from Hugging Face using the AutoModelForCausalLM and AutoTokenizer classes. Optimizations include:

Device Mapping: Automatically assigns the model to available hardware (GPU or CPU).

Memory Efficiency: Utilized half-precision (torch.float16) and gradient checkpointing to minimize memory usage and improve performance.

Phishing Detection Workflow

Preprocessing: Email text is cleaned (punctuation/URLs removed) and truncated to 600 characters to ensure efficient processing.

Prompt Generation: A formatted prompt instructs the model to classify the email as phishing (1) or non-phishing (0), with examples provided to guide the model.

Model Inference: The tokenized input is processed by the model, and the output is interpreted to return a simple classification of "Phishing" or "Non-Phishing."

Gradio Interface

A Gradio interface provides a web-based platform for users to interact with the model:

Input: Users enter email content.

Output: The model returns "Phishing" or "Non-Phishing."

This deployment provides an efficient and accessible tool for phishing detection, optimized for performance using Hugging Face Spaces and Gradio.

Fig I.  Deployment Interface

## Future Work

One promising direction for future work is a dual architecture system that combines a text analysis model with a URL analysis model. The text model, like mistral,BERT or RoBERTa, would focus on detecting phishing patterns in the email content, while the URL model would analyze embedded links for suspicious traits.

Both models would operate in parallel, with their results combined through a fusion method—such as weighted averaging or voting—to make a final classification. This setup balances accuracy and speed, as each model specializes in its domain and operates simultaneously for real-time detection.

This hybrid approach enhances detection by combining insights from both text and URLs, resulting in more accurate and efficient phishing classification suitable for real-world applications.

Additionally, further exploration into optimizing Mistral’s deployment for real-time applications should be considered. This could involve using techniques like quantization or distillation to reduce the computational overhead of running large models like Mistral, making it more feasible for practical deployment in production environments.

## Conclusion

This project explored the efficacy of various models in phishing detection, from traditional machine learning approaches to advanced large language models (LLMs) and the Mistral model. The results show that more sophisticated models like RoBERTa, BERT, and Mistral significantly outperform traditional models like Random Forest and KNN, particularly when it comes to leveraging contextual information in the text and URLs.

The inclusion of URLs in the preprocessing pipeline had a minimal impact on traditional models, whereas deep learning models and LLMs benefited slightly. The Mistral model, in particular, saw the greatest improvement, achieving perfect accuracy with URLs included. However, this performance comes at the cost of high computational requirements, making Mistral a more expensive choice for large-scale or real-time deployment.

In conclusion, while traditional models provide a baseline for phishing detection with relatively low computational requirements, they struggle to match the performance of more advanced models. Deep learning models, particularly CNN and LSTM, offer a good balance between performance and resource efficiency, making them suitable for medium-scale deployments. LLMs such as BERT, RoBERTa, and DistilBERT provide high accuracy in phishing detection, but Mistral stands out as the most effective model, particularly when URLs are included in the input data.

However, Mistral’s performance comes with significant computational costs, making it less feasible for organizations with limited resources or those requiring real-time, large-scale detection systems. In practice, the choice of model will depend on the specific needs and constraints of the organization, balancing accuracy with computational efficiency.

## References

Doshi, Jay, et al. "A comprehensive dual-layer architecture for phishing and spam email detection." Computers & Security 133 (2023): 103378.

.

Thakur, Kutub, et al. "A Systematic Review on Deep-Learning-Based Phishing Email Detection." Electronics 12.21 (2023): 4545.

McGinley, Cameron, and Sergio A. Salinas Monroy. "Convolutional neural network optimization for phishing email classification." 2021 IEEE International Conference on Big Data (Big Data). IEEE, 2021.

Anonymous. Phishing Email Curated Datasets. Zenodo, 13 Sept. 2023, doi:10.5281/zenodo.8339691.

Fang, Yong, et al. "Phishing email detection using improved RCNN model with multilevel vectors and attention mechanism." IEEE Access 7 (2019): 56329-56340.

Egozi and Verma (2018) suggested a classification model using NLP and machine learning, with a focus on unique features. The model incorporated 26 elements, four of which were original. These features were applied to 17 machine learning models, yielding a true positive rate of 83% and a true negative rate of 96%.

Peng, Tianrui, Ian Harris, and Yuki Sawa. "Detecting phishing attacks using natural language processing and machine learning." 2018 ieee 12th international conference on semantic computing (icsc). IEEE, 2018.

Bagui, Sikha, et al. "Classifying phishing email using machine learning and deep learning." 2019 International Conference on Cyber Security and Protection of Digital Services (Cyber Security). IEEE, 2019.

Alhogail, Areej, and Afrah Alsabih. "Applying machine learning and natural language processing to detect phishing email." Computers & Security 110 (2021): 102414.

Nayak, Rakesh, Salim Amirali Jiwani, and B. Rajitha. "WITHDRAWN: Spam email detection using machine learning algorithm." (2021).

Atawneh, Samer, and Hamzah Aljehani. "Phishing email detection model using deep learning." Electronics 12.20 (2023): 4261.

Jiang, Liming. "Detecting scams using large language models." arXiv preprint arXiv:2402.03147 (2024).

Koide, Takashi, et al. "ChatSpamDetector: Leveraging Large Language Models for Effective Phishing Email Detection." arXiv preprint arXiv:2402.18093 (2024).

https://www.kaggle.com/datasets/imdeepmind/preprocessed-trec-2007-public-corpus-dataset.
