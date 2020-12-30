projects = [
    {
        "top5": true,
        "title": "Meta-Learning in Manipulation",
        "team": "Team Grasp",
        "video": "https://youtu.be/cp1g_qhH-Go",
        "report": "https://drive.google.com/file/d/1K_surkWundLtXzsFKgQorjbpayfzid0x/view?usp=sharing",
        "summary": "Reinforcement learning (RL) is a powerful learning technique that enables agents to learn useful\u00a0policies by interacting directly with the environment.\u00a0 While this training approach doesn\u2019t require labeled data, it is plagued with convergence issues and is highly sample inefficient. The learned policies are often very specific to the task at hand and are not generalizable to similar task spaces. Metal-ReinforcementLearning is a one strategy that can mitigate these issues, enabling robots to acquire new skills much more quickly. Often described as \u201clearning how to learn\u201d, it allows agents to leverage prior experience much more effectively.\u00a0 Recently published papers in Meta learning show impressive speed and sample efficiency improvements over traditional methods of relearning the task for slight variations in the task objectives.\u00a0 A concern with these meta-learning methods was that their success was only achieved on relatively small modifications to the initial task.\u00a0 Another Concern in RL is reproducibility and lack of standardization of the metrics and approaches, which can lead to wide variations in reported vs observed performances. To alleviate that, benchmarking frameworks have been proposed that establish a common ground for a fair comparison between approaches. In this work, we aim to utilize the task variety proposed by Meta-World and compare PPO, a vanilla policy gradient algorithm, vs their meta-learning counterparts\u00a0 (MAML\u00a0 and\u00a0 Reptile). \u00a0 The\u00a0 objective\u00a0 is\u00a0 to\u00a0 verify\u00a0 the\u00a0 magnitude of success of meta-learning algorithms over the vanilla variants, and test them on a variety of complicated tasks to test their limits in responding to the size of task variations. We also introduce a new technique - Multi-headed Reptile,which addresses some of the shortcomings of both these meta learning techniques. Additionally, we propose a method to speed up the training process to compensate for the general lack of vectorizability of some parts of reinforcement learning algorithms.",
        "pic": "Team Grasp.png"
    },
    {
        "top5": true,
        "title": "Multi-SAP Adversarial Defense for Deep Neural Networks",
        "team": "Enthu-Cutlets",
        "video": "https://youtu.be/GOS71WOBSN8",
        "report": "https://drive.google.com/file/d/1o3WRdTOIFwEZ8YRRGa5aY-dkXk59wibM/view?usp=sharing",
        "summary": "Deep learning models have gained immense popularity for machine learning tasks such as image classification and natural language processing due to their high expressibility. However, they are vulnerable to adversarial samples - perturbed samples that are imperceptible to a human, but can cause the deep learning model to give incorrect predictions with a high confidence. This limitation has been a major deterrent in the deployment of deep learning algorithms in production, specifically in security critical systems. In this project, we followed a game theoretic approach to implement a novel defense strategy, that combines multiple Stochastic Activation Pruning with adversarial training. Our defense accuracy outperforms that of PGD adversarial training, which is known to be the one of the best defenses against several L∞ attacks, by about 6-7%. We are hopeful that our defense strategy can withstand strong attacks leading to more robust deep neural network models.",
        "pic": "Enthu-Cutlets.png"
    },
    {
        "top5": true,
        "title": "Game Theory For Adversarial Attacks And Defenses",
        "team": "Team IDL.dll",
        "video": "https://youtu.be/MeBxlQ-n6e4",
        "report": "https://drive.google.com/file/d/14N2EqPzTnymytV9Izg6KsVfSNy0yyHVn/view?usp=sharing",
        "summary": "Adversarial attacks can generate adversarial inputs by applying small but intentionally worst-case perturbations to samples from the data set, which leads to even state-of-the-art deep neural networks outputting an incorrect answer with high confidence. Hence, some adversarial defense techniques are developed to improve the security and robustness of the models and avoid them being attacked. Gradually, a game-like competition between attackers and defenders formed, in which both players would attempt to play their best strategies against each other while maximizing their own payoffs. To solve the game, each player would choose an optimal strategy against the opponent based on the prediction of the opponent\u2019s strategy choice. In this work, we are on the defensive side to apply game-theoretic approaches on adversarial training the models. We generate adversarial training samples by assuming attackers would do a fast gradient sign method (FGSM) white-box attack. Using both the raw data set and generated data set to train the models, we reduce the test data set error while the models being attacked on the CIFAR-10 data set. Our experimental results indicate that the adversarial training method can effectively improve the robustness of deep-learning neural networks. Our proposed method aims to use diverse networks to simulate the mixed strategy thus to reduce the success rate of attack",
        "pic": "Team IDL.dll.png"
    },
    {
        "top5": true,
        "title": "Multi-modal Image Cartoonization",
        "team": "Sailor Moon",
        "video": "https://youtu.be/irD-Unhc-Ms",
        "report": "https://drive.google.com/file/d/1W3U_NQCGpBt71AVUI43y1iNrr2bcpHT7/view?usp=sharing",
        "summary": "In this project, we study the problem of multi-modal image cartoonization. Current image cartoonization methods are single modal models, which restrict the scalability to multi-modal. We proposed a model based on StarGAN and StyleGAN to solve this challenging problem. For the midterm checkpoint, we implemented 3 baseline models to learn the problem and limitation of current models and show our initial experiments of them. Code available at: https://github.com/Hhhhhhhhhhao/image-cartoonization",
        "pic": "Sailor Moon.png"
    },
    {
        "top5": true,
        "title": "A Walk to Remember: Trajectory Prediction using Recurrent Networks \ud83d\udeb6\u200d\u2642\ufe0f",
        "team": "Team Rocket",
        "video": "https://youtu.be/_HqzPlvgugs",
        "report": "https://drive.google.com/file/d/1BmfT-pop_mVnt-vApY0uitXSi8iQsqCk/view?usp=sharing",
        "summary": "Trajectory prediction is an important problem that has applications in autonomous driving, safety monitoring, object tracking, robotic planning, and so on. Current state-of-the-art methods focus on either single or multiple trajectory prediction using generative models (such as Generative Adversarial Networks), as well as discriminative models. In this project we outline several methods that achieve competitive results, highlight the key contributions of their approaches, and justify our choice of adopting a discriminative approach. Starting with the Social-LSTM model as our baseline, we propose several possible extensions to improve the quality and length of predicted trajectories.",
        "pic": "Team Rocket.png"
    },
    {
        "title": "Adversarial Defense Using Generative Adversarial Networks",
        "team": "StarsAlign",
        "video": "https://www.youtube.com/watch?v=EhtTznS63cE",
        "report": "https://drive.google.com/file/d/19oa8xKGiRoPrqaZ4jylXBAI3gqDkXGWz/view?usp=sharing",
        "summary": "Neural networks are known to be vulnerable to adversarial attacks - carefully modified inputs that are imperceptibly similar to natural inputs, that cause the network to fail catastrophically at the intended task. In our project, we aim to make targeted classifier models more robust to adversarial attacks. We train a generative model in the context of a generative adversarial network, and use this to generate perturbations that can be added to the input image to counter the perturbation added by the attacker. We also explore defense by generating images that are as close as possible to the input adversarial image. We implement our own attacks and train our own baselines to ensure uniform comparison.",
        "pic": "StarsAlign.png"
    },
    {
        "title": "Generating Accurate Human Face Sketches from Text Descriptions",
        "team": "Optimax",
        "video": "https://youtu.be/ryUwl8ga1y8",
        "report": "https://drive.google.com/file/d/1Q3zDpPcPHwKXBG9ePnzAh7wgmvz45MO2/view?usp=sharing",
        "summary": "Drawing a face for a suspect just based on the descriptions of the eyewitnesses is a difficult task.It requires professional skills and rich experience. It also requires a lot of time. However, with a well trained text-to-face model, anyone could directly generate photo-realistic faces of suspects based on the descriptions of eyewitnesses quickly. Most previous research on sketch generation assume that the original photo is available, which are usually unavailable from description of suspects. Since text-to-face synthesis is a sub-domain of text-to-image synthesis, there are only a few research focusing in this sub-domain (although it has more relative values in the public safety domain). Here, we use AttentionGAN to generate sketches from the text descriptions. Our AttentionGAN successfully captures the text semantics into sketch image domain.",
        "pic": "Optimax.png"
    },
    {
        "title": "Makeup Transfer with Pose and Expression Robust",
        "team": "chanzysb",
        "video": "https://youtu.be/-6oH5pID8wo",
        "report": "https://drive.google.com/file/d/1V_ZuJgbrOwNlB40pzPJz6ZyF9qwLVgFt/view?usp=sharing",
        "summary": "In our project, we focus on making pose and expression robust makeup transfer. We extract the makeup from a reference image and apply it to the source image. There are mainly three steps to reach the goal. Firstly, we extract the makeup from the reference image using Makeup Distill Network model as two makeup matrices. Next, we use Attentive Makeup Morphing model to find the change of a pixel in the source image from the reference image. Finally, we apply Makeup Apply Network to perform the makeup transfer. By applying these methods, we can improve the shortcomings of existing makeup transfer models that they are not suitable for large pose and expression variation and our model is pose and expression robust.",
        "pic": "chanzysb.png"
    },
    {
        "title": "AlphaQuantum: Mastering the game of Quantum Tic Tac Toe without human knowledge",
        "team": "Erwin Schr\u00f6dinger\u2019s Neuron",
        "video": "https://youtu.be/VLC6ztza2k8",
        "report": "https://drive.google.com/file/d/19oHWo51PynQwI93GUkkv8OOZiBpMfXfL/view?usp=sharing",
        "summary": "Reinforcement learning has been widely used for artificial intelligence in playing games and proven having the ability to defeat human players in many cases. Quantum Tic-Tac-Toe (QTTT) game adds the idea of superposition to the classic Tic-Tac-Toe, which implies a huge piece arrangement space up to 10^8. The projects implements 2 reinforcement learnining algorithms: temporal difference learning and AlphaZero. Both cases policies are learnt through self-play. Initial experiment result shows that AlphaZero can achieve a professioncy equivalent to the medium level AI provided by the Qttt game on the google app store.",
        "pic": "Erwin.png"
    },
    {
        "title": "Deep Contextualized Term Weighting for Ad-Hoc Document Retrieval",
        "team": "thinkingface\ud83e\udd14",
        "video": "https://youtu.be/IdrXnbTFs9M",
        "report": "https://drive.google.com/file/d/1KHcbQtFiNKBuS0dTTOQhO8bYODDPARSC/view?usp=sharing",
        "summary": "Bag-of-words representation with term-frequency based term weighting has been used for a long time in modern search engines. Although it is powerful in modeling document-query term interaction, its power is restricted by shallow frequency-based term weighting scheme. Recently, there has been some successful researches conducted using BERT to estimate term weights in a contextualized fashion. In our project, we propose to experiment models that can improve upon those original researches.",
        "pic": "thinkingface.png"
    },
    {
        "title": "Quantum Tic Tac Toe",
        "team": "Tic Tac Toe",
        "video": "https://youtu.be/WXiUDKCsXa0",
        "report": "https://drive.google.com/file/d/1926VWIuCDjJgjmt8mCCO2UHFJgPWmWWe/view?usp=sharing",
        "summary": "Leverage deep learning techniques to solve different version of Tic Tac Toe (normal, and different version of quantumness) and compare their performance. Research how different types of learnings perform in different complexities of games)",
        "pic": "Tic Tac Toe.png"
    },
    {
        "title": "Game Theoretic Approaches to Adversarial Attacks and Defenses",
        "team": "Epsilon",
        "video": "https://www.youtube.com/watch?v=yQOE1eCtCTo",
        "report": "https://drive.google.com/file/d/104JL1v3GJZ1crmk5Kx3B64zqchEC4siV/view?usp=sharing",
        "summary": "Deep neural networks are vulnerable to attacks called adversarial attacks in which, data sample that are imperceptibly different from a natural data samples are input to a neural network such that it forces the network to predict a class different from the original class that the samples belong to. The idea is to approach adversarial attacks and defenses through a game-theoretic perspective, thus coming up with robust defenses for the model under consideration. In our approach, we tried to implement models with good classification accuracy and subject them to two of the most popular attacks Fast Gradient Sign method (FGSM) and Projected Gradient Descent (PGD).Ideally, the attacks should be assumed to be belonging to the class of White box attacks, because the defense assumes the worst case scenario. The game between the attacker and defender is modeled as a multi-move game with zero-sum reward where the defender has the last move advantage. In accordance with this, we defend against the \u201cattack\u201d after we get the image from the attacker. In our pipeline we combine image cut-out and in-painting on adversarial generated images with the aim of reducing the effect of adversarial perturbation on the classification accuracy.",
        "pic": "Epsilon.jpg"
    },
    {
        "title": "Long-term Joint Trajectory and Action Prediction with Intention Recognition",
        "team": "Precognition",
        "video": "https://youtu.be/eNYatmMjGcU",
        "report": "https://drive.google.com/file/d/1L_9TZ9vtCwK2Azu5OILtu-6iu9Cyh9al/view?usp=sharing",
        "summary": "This paper studies the problem of long-term prediction of multiple possible future paths of people as they move through various visual scenes. We make two main contributions. The first contribution is a new dataset, which extends previous common prediction horizon by two-folds and allows for comprehensive scene analysis. This provides the first benchmark for quantitative evaluation of the models to predict long-term trajectories and actions with rich environmental constraints. The second contribution is a new model that utilizes intention forecasting based on scene graphs and social interactions for long-term joint trajectory and action prediction. We refer to our model as IntentNet. We show that our model achieves the best results on our dataset.",
        "pic": "Precognition.png"
    },
    {
        "title": "Ensemble-based adversarial defense",
        "team": "MIIS Big Four",
        "video": "https://www.youtube.com/watch?v=DbSUzBIDKtY",
        "report": "https://drive.google.com/file/d/1W3U_NQCGpBt71AVUI43y1iNrr2bcpHT7/view?usp=sharing",
        "summary": "Exploring the ensemble-based defense methods to defend adversarial attacks.",
        "pic": "MIIS Big Four.jpg"
    },
    {
        "title": "Study on Layer-wise Divergence Control Mechanism against Adversarial Attacks",
        "team": "BestHotpotPittsburgh",
        "video": "https://youtu.be/LCQZ-Awnn6g",
        "report": "https://drive.google.com/file/d/1OU0gitpWJnoKI7dxs7Mi-4NfCd5h8fm2/view?usp=sharing",
        "summary": "With the wide adoptions of deep learning models in many aspects in our lives, the reliability and security of deep learning models is gaining more and more attention from researchers, and many innovative network architectures were proposed to address possible defense mechanisms against adversarial attacks. In this project, we proposed a Layer-wise Divergence Control (LDC) mechanism in order to defend against adversarial attacks. We evaluated its performance under the context of white-box attacks, comparing to the state-of-the-art robust defense model architecture, TRADES. The Layer-wise Divergence Control mechanism we proposed can achieve similar performance to TRADES against PGD INF attacks. In terms of PGD L2 attacks, TRADES significantly outperforms our mechanism. However, we still believe we can achieve a better defense by exploring different combination of hyper-parameters and divergence functions for our LDC mechanism, and we will continue working to improve its performance. ",
        "pic": "BestHotpotPittsburgh.png"
    },
    {
        "title": "Tackle Multimodal QA with Multi-Stage Information Fusion",
        "team": "Sunday Dues",
        "video": "https://www.youtube.com/watch?v=wsoCuHaYNCs",
        "report": "https://drive.google.com/file/d/1pUhz558hDBoMQj67a_TKJvVfDPkY689Y/view?usp=sharing",
        "summary": "Visual Question Answering (VQA) has wide applications in real life and is one of the benchmarks for artificial intelligence systems. In this paper, we propose a new solution to a related task, MemexQA, in which photos from personal albums as well as photo metadata are given to answer a multiple-choice question about the information presented in the albums such as locations, dates, events and people. Specifically, we tackle this task by improving the information fusion process of a baseline LSTM model by introducing a multi-stage fusion method for features of different modalities.",
        "pic": "Sunday Dues.jpg"
    },
    {
        "title": "Triple Defense Against Adversarial Attacks",
        "team": "\u00a0Up!Its8AM",
        "video": "https://youtu.be/ZtKidQgfvQk",
        "report": "https://drive.google.com/file/d/1G6Fel5ABZlBQ6jsGYqufYtzZ2TBbD1D0/view?usp=sharing",
        "summary": "The recent development in the adversarial attacks can be maliciously used to cause a perfectly performing Deep Learning model to make incorrect predictions. This is particularly harmful in models where a misclassification becomes a matter of life and death. To address this problem, we are proposing a Triple Defense System, by creating an Ensemble Model of ShuffleNet models, where we are adding randomness at all the three levels of training, model selection and inference. This will lead to a Robust architecture against any type of adversarial attacks. To check the performance of our Triple Defense System, we have attacked the model with both Fast Gradient Method and Projected Gradient Descent attacks( L2 and Linf norm) with different adversarial noises. Our experiments are based on varying different parameters like noise, number of models at both training and inference stage and finding the best hyperparameters to provide defense against strong malicious attacks. ",
        "pic": "Up!Its8AM.png"
    },
    {
        "title": "Real Time Road Scenes Semantic Segmentation",
        "team": "YixiuYixiu",
        "video": "https://youtu.be/FEsIlFf55bM",
        "report": "https://drive.google.com/file/d/1VxXoE7xBcKM7F1DoZeQelxpa_zEAOvRu/view?usp=sharing",
        "summary": "Computer vision is a critical part of automated vehicle perception systems, responsible for the detection of vehicles, traffic signs, lanes, pedestrians and obstacles.The application of semantic segmentation in autonomous driving is quite common.Currently, many mainstream segmentation models have high accuracies, but theframe rates are low. In driverless scenarios, the model must be real-time, especially in high-speed scenarios, which requires higher inference speed of the model. In this project, we aim at detecting the surrounding environment of vehicles via processing real time videos with our optimized models. For this purpose, we used simple DeconvNet (four layers) as our baseline model, and improved simple DeconvNet performance while using and optimizing other models, including U-Net, FCN-8, a combined model of Vgg16, FCN-8, FCN-16 and FCN-32(DAG). We calculated the intersection over union (IoU) for each object and mean IoU while paying attention to model speed, summarized and compared model parameters. As a result, DAG(Vgg16 + FCN8s + FcN16s + FCN32s) with data augmentation performs best,achieving 89.3 percent global accuracy and 66.19 percent mean IOU.",
        "pic": "YixiuYixiu.png"
    },
    {
        "title": "Exploring Feature extraction process in WSJ ASR Task",
        "team": "PSInet",
        "video": "https://youtu.be/UA0VCsic9fo",
        "report": "https://drive.google.com/file/d/1xPSyoV9X2Yrg65JmEfGVFQQAPEdYhRV8/view?usp=sharing",
        "summary": "This project focuses on the feature exploration task of finding new ways to process source signals in ASR systems. The current feature extraction process has been in place for around 20 years, but current advances in Deep Learning have led to the possibility that new or modified feature extraction methods can improve the performance of ASR systems. We explore 2 main avenues for this process, 1 is to modify the feature extraction process from the Kaldi pre-processor used in ESPnet and the other is to replace the pre-processor and the other we consider is to use a neural approach instead of the typical Kaldi Preprocessing.",
        "pic": "PSInet.png"
    },
    {
        "title": "Disparity Estimation in Stereo Images",
        "team": "g4dn.Xlarge insufficient capacity",
        "video": "https://youtu.be/RJvEDMKQ5GA",
        "report": "https://drive.google.com/file/d/1ImUaMc9U6PABpNnNwSmU2VwTazJUY4EK/view?usp=sharing",
        "summary": "Computer vision applications such as autonomous driving, 3D model reconstruction, and object detection and recognition rely heavily on depth estimation from stereo images. Recent state-of-the art work has shown that depth estimation can be formulated as a supervised learning task using convolutional neural networks (CNNs), given a stereo pair of images as input. There have been numerous recent proposed methods to tackle this challenge, but perhaps the most fundamental and well-known is the Pyramid Stereo Matching Network (PSMNet). In this project we will investigate the various methods used by PSMNet, and augment their findings with some additional methods of our own. These methods include, but are not limited to, architecture modifications, parameter reduction techniques, and disparity estimation from infrared (IR) images. Ultimately, the goal of our project is to provide a contribution to the scientific community by enabling others to use the results of our findings in their own projects.",
        "pic": "g4dn.Xlarge insufficient capacity.png"
    },
    {
        "title": "Whole-Document Embedding and Classification for the Clinical Domain",
        "team": "ICD Light",
        "video": "https://youtu.be/e6VYVbTY6wk",
        "report": "https://drive.google.com/file/d/1KuKJWyKL-i2-HupMWBMUxUl8iulrOUK5/view?usp=sharing",
        "summary": "Using novel pre-training of Transformer encoders, this project tackles whole-document embedding for the clinical domain. Additionally, we propose a fine-tuning process on electronic healthcare records for transformer models and a novel medical coding benchmark task. We release our best-performing encoder model and suggest future investigation with regard to the natural language tasks in this domain.",
        "pic": "ICD Light.png"
    },
    {
        "title": "Stereo Analysis using Low Texture Thermal Data",
        "team": "Sleep Learning",
        "video": "https://youtu.be/ZJExw2yjtJI",
        "report": "https://drive.google.com/file/d/1meQRTbGtxjmmn7oSIfhFiVJ3LDBfKkzV/view?usp=sharing",
        "summary": "Infrared Images (IR) images are known to be robust to ambient illumination, fog and incoming car headlights. These features should make them favorable for autonomous driving applications, however, they are less frequently used due to their noisy and texture less nature. In addition to this, IR Cameras embed the thermal component emitted by the objects which give them better visibility in extreme weather conditions or in low light scenes, and hence a better performance as compared to RGB cameras, in such situations. In this report, we have described the methods employed to estimate disparity from low texture, noisy IR data. For the baseline architectures, we have selected AANet (Adaptive Aggregation Network)and GA-Net (Guided Aggregation Net), which are state of the art architectures for disparity estimation using RGB stereo images. These models employ fewer 3Dconvolutional layers, which enables operation at real time speeds. In experiments,we show how these models perform on IR stereo data, after extensive tuning of parameters. For training and evaluation, we use the CATS (A Color And Thermal Stereo) dataset and yet to be published, CMU dataset. Finally we discuss the shortcomings of these models and describe our proposed method which improves upon the baseline architectures. This field of estimating disparity using IR images,remains unexplored, and we aim to solve this problem. The link to our GitHub repository, https://github.com/karan2808/STEALTH.git.",
        "pic": "Sleep Learning.png"
    },
    {
        "title": "Semi-supervised learning for better pedestrian detection in thermal images",
        "team": "Endless Deep",
        "video": "https://youtu.be/uHPqJcPMQDw",
        "report": "https://drive.google.com/file/d/1SSWwAOS04VDqUeSrs6hrdaNefBIYWp33/view?usp=sharing",
        "summary": "The accuracy of pedestrian detection can ensure safety in autonomous driving. While deep learning for pedestrian detection in RGB images has achieved good performance in supervised deep learning, the learned model tempts to overfit a specific dataset. Another problem is that RGB cameras can be influenced by lightning or camera range, making the model hard to be applied in realistic scenes. Thermal images have the advantages that they are invariant with lighting and have a larger detection range. However, training a new deep model requires a large dataset, and collecting many annotated data is expensive and time-consuming. To reduce this bottleneck, we utilize semi-supervised transfer learning, including active-learning and self-training, to train on a dataset with a limited number of labels, to reduce the labeling cost. The semi-supervised learning algorithm usually has lower performance than supervised training on a large dataset. We creatively use the ensemble-learning technique in semi-supervised learning to ensemble the knowledge learned from different methods to improve the performance of semi-supervised learning.",
        "pic": "Endless Deep.jpg"
    },
    {
        "title": "Optimization of ESPnet Feature Extraction",
        "team": "Gold Miner",
        "video": "https://youtu.be/7SPblpgVtr8",
        "report": "https://drive.google.com/file/d/16TOWq-FjV5cga5oH1baZ-H1KaRPAv3iJ/view?usp=sharing",
        "summary": "Automatic speech recognition (ASR) studies converting spoken words into text by machine. It has been a field of research since the late 1950s, and researchers have developed a mature speech recognition pipeline consists of feature extraction, acoustic modeling, lexicon, and language modeling. Our project will focus on feature extraction, which converts the waveform speech signal to a set of feature vectors via different types of transformation. The current speech feature configuration (log mel filterbank) was fixed for more than 20 years and the old standards may not be optimal in the deep learning era. Therefore, our team will explore novel ways of feature extraction to improve the performance of speech recognition.",
        "pic": "Gold Miner.jpg"
    },
    {
        "title": "Deep Learning Approaches to Antibiotic Classification",
        "team": "Mariana Trench\u00a0",
        "video": "https://youtu.be/j5TbS0NmN1M",
        "report": "https://drive.google.com/file/d/1wzoaBjtLhqSTVvwOg1tuCWCttUjCJl5C/view?usp=sharing",
        "summary": "We proposed two approaches to perform antibiotic molecule discovery and classification: 1. Simple Graph Neural Network: Applying graph neural network on atom/bond level features. 2. Fingerprint Vector Model: Applying conventional neural network on molecular level features (molecular fingerprint). We prepared a dataset of 2335 molecules with labels, and extracted atom/bond/molecular features from the dataset. Baseline models are built for both approaches. While graph neural network model is still in training, Multi-layer perceptron model applied to molecular fingerprints is able to achieve a 85.8% testing accuracy. However, we discovered that the dataset is unbalanced, with 95% of samples in one label. Since the data is collected by biochemistry experiments, the amount of data is limited. We decide to use data augmentation techniques such as SMOTE (Synthetic Minority Oversampling Technique) and ensembling in our future work. In the rest of semester, our goal is to improve current situation of limited number of data and enhance performance of current models.",
        "pic": "Mariana Trench.png"
    },
    {
        "title": "A GPT2 based Limerick Generator",
        "team": "Youshen",
        "video": "https://youtu.be/b16xOLpG_p0",
        "report": "https://drive.google.com/file/d/1BORroctAKMLcZmmnmxDZjQwPFYDA8yJ2/view?usp=sharing",
        "summary": "Poetry is a written art form that adapts standard language rules to create a richer piece of text. There are many different styles of poetry that adhere to different rules or structures, for example Haiku. Deep learning is the state of the art tool for automatic text generation. Current state of the art like GPT-2, uses a transformer architecture and is capable of generating text with arbitrary length and context. Rhyming is a unique feature of poetry that we thought would be a new and challenging generation task for a GPT-2 model. We have trained the GPT-2 model on a limerick corpus of 90,000 from oedilf.com. We developed several automatic evaluation metrics such as rhyming coherence, subject co-reference and nonsense evaluation. We also developed a website where we asked people if they could distinguish Human limericks from our training set vs. novel generated limericks from our model. Preliminary results show that we were able to fool the human 76 times or ~17% of the responses. Based on the rhyming structure of limericks (AABBA) we have named and will refer to our model as AiBBA.",
        "pic": "Youshen.jpg"
    },
    {
        "title": "Anchor-free Object Detection using Multi-Level Feature Pyramids",
        "team": "Peaky Blinders",
        "video": "https://youtu.be/2-GGyW59CLw",
        "report": "https://drive.google.com/file/d/116Mnz-RapOf3RT0dnZ50SrNQX-JpMuGW/view?usp=sharing",
        "summary": "Object detection is breaking into a wide range of industries, with use cases rang- ing from personal security to automated vehicle safety. While object detection techniques have improved massively over the past decade, most state of the art detectors rely on pre-defined anchor boxes and 2-stage region proposal networks. In this project, we try to implement an anchor-free, region proposal-free detector which makes use of multi-level feature pyramid network for greater localization and classification accuracy. For the mid-term report, we have implemented an anchor-based FPN network as a baseline which is able to achieve an mAP value of 0.75. We are now working towards improving this performance using multi-level FPNs for the final submission.Andrew IDs:dvashish, bmullick, shayeres, zhenweilPeer Rating Form:",
        "pic": "Peaky Blinders.png"
    },
    {
        "title": "Semantic Segmentation for Urban-Scene Images",
        "team": "DL Sailor Moon",
        "video": "https://youtu.be/701ZJbv6TwI",
        "report": "https://drive.google.com/file/d/1hSaDJ_vvPy047mdopIi8bHY9_iKP4SNQ/view?usp=sharing",
        "summary": "Urban-scene Image segmentation is an important and trending topic in computer vision with wide use cases like autonomous driving [1]. Starting with the break-through work of Long et al. [2] that introduces Fully Convolutional Networks (FCNs), the development of novel architectures and practical uses of neural net-works in semantic segmentation has been expedited in the recent 5 years. Aside from seeking solutions in general model design for information shrinkage due to pooling [], urban-scene image itself has intrinsic features like positional patterns[3]. Our project seeks an advanced and integrated solution that specifically targets urban-scene image semantic segmentation among the most novel approaches in the current field. We re-implemented the cutting edge model DeepLabv3+[cite] withResNet-101[cite] backbone. Based upon DeepLabv3+, we incorporated Height-Driven Attention Net [3] to account the vertical pixel information distribution in urban-scene image tasks. To boost up model efficiency and performance, we further explored the Atrous Spatial Pooling(ASP) layer in DeepLabv3+ and infused a computational-efficient variation called \"Waterfall\" Atrous Spatial Pooling(WASP)[4] architecture in our model. We found that our two-step integrated model im-proves the mean Intersection-Over-Union(mIoU) score gradually from the baseline model. In addition, we demonstrated the improvement of model efficiency with help of WASP in term of computational times and parameter reduction.",
        "pic": "DL Sailor Moon.png"
    },
    {
        "title": "COVID-GAN - Augmenting COVID-19 Diagnostic Data using Hybrid VAE-GANs",
        "team": "The Cerebral Catz\u00a0",
        "video": "https://youtu.be/gERzi7Wd2Y4",
        "report": "https://drive.google.com/file/d/1uUsnXxLM5Zupq_BiS8UsJT0hFNNZVqwb/view?usp=sharing",
        "summary": "COVID-19 had become the new norm since the pandemic. While the number of cases are rising on a daily basis, there is a steady need to keep up rising number of cases and automate the diagnosis of COVID-19, offloading some of the work off the front-line workers. It is a known fact that medical image data suffers from skewness due to the limited availability of positive COVID-19 patient data. Deep learning models rely heavily on the uniformity of the data to produce a good result and hence often overfit on highly skewed data. Increasing the number of training samples has been shown to provide better classification accuracy and more generalizability. Due to strict protocols for medical data collection, it is not feasible to collect additional data for training. In a scenario like this, the best approach is to perform data augmentation on the available training samples to reduce class imbalance and generate more samples. Many approaches to data augmentation have been proposed, from basic image transformations to using a Generative network to create new images. In this work, we apply a combination of Variational Autoencoders and Generative Adversarial Networks to augment the COVID-19 data on which a CNN-based classifier is trained to achieve a better performance.",
        "pic": "The Cerebral Catz.png"
    },
    {
        "title": "Adversarial attacks defenses via Modeled random noises",
        "team": "AT Field",
        "video": "https://www.youtube.com/watch?v=Ysx2nYwx4Ok",
        "report": "https://drive.google.com/file/d/18YPO-9RbmHbkr9WPP5-fTvYMN2NZ5f62/view?usp=sharing",
        "summary": "Neural networks are known to be vulnerable to adversarial examples: natural\u00a0inputs with carefully constructed perturbations may render them to classify input\u00a0incorrectly. In this project, we conduct experiments to illustrate such phenomenon.\u00a0We also extend Randomized Smoothing defense by using targeted Gaussian Noise\u00a0preferring the direction of first-order gradients of the network.",
        "pic": "AT Field.png"
    },
    {
        "title": "Bitcoin Price Prediction using Sentiment and Historical Price",
        "team": "Team 32",
        "video": "https://youtu.be/Nr_kUkE_L-o",
        "report": "https://drive.google.com/file/d/1dHwC6PTIGO7WQnhj0zU33TMvH0JjeqdJ/view?usp=sharing",
        "summary": "Bitcoin is one of the first digital currencies that were first introduced around 2008. The decentralized currency gained huge popularity in the past decade and it is now trading around USD 10000 per coin. Speculations around its future upside potential have attracted a large number of individual and institutional investors despite the price volatility. Therefore, price prediction for bitcoin has become a relevant but challenging task. In this project, we will be utilizing sentiment analysis and historical price data to maximize the accuracy of price prediction of bitcoin.",
        "pic": "Team 32.png"
    },
    {
        "title": "Unsupervised Anomaly Detection in Electrocardiograph Time Series Data Using Variational Recurrent Autoencoders with Attention, and Transformer",
        "team": "Transformed Attention",
        "video": "https://youtu.be/ivJgVVaALY0",
        "report": "https://drive.google.com/file/d/11nSlSR0FvvUTLL-3yZV-9Zc8FYJuTD_I/view?usp=sharing",
        "summary": "Time series data a long existed in various fields in our daily life. It is hard to extract the pattern of the data by human effort. Since computer intelligence is expected to perform more sensitive than human beings, deep learning algorithms, especially data pattern detection have been applied to detect anomaly patterns from complex systems, such as heartbeat signals. However, a significant amount of approaches based on supervised machine learning models that require (big) labelled data-set. In this project, we are expected to detect the pattern of heartbeat with an optimized Auto-Encoder model. Our contributions can be divided as three steps. Firstly, we established two baseline models (MLP and LSTM auto-encoder) on MNIST dataset. The result shows good sensitivity to distinguish from different data. Secondly, we add attention mechanisms to LSTM decoder to improve theperformance. Finally, we use multi-head attentional transformer model instead ofLSTM auto-encoder to explore the possibilities of replacement. We can see a very clear loss threshold between the normal and abnormal data.",
        "pic": "Transformed Attention.png"
    },
    {
        "title": "End to End Question Answering",
        "team": "TBD - Team 34",
        "video": "https://youtu.be/mBTJcJhdrjs",
        "report": "https://drive.google.com/file/d/1iJ9dMX5EhonCMc4kynetvsZZXKPYEfpy/view?usp=sharing",
        "summary": "Question answering is a popular research topic in natural language processing, and a widely used application in real world scenarios nowadays. In this paper we propose a method based on BERT, which is a recurrent neural network composed of an encoder-decoder architecture, along with multihead self-attention mechanism, which is known as the \"Transformer\".Several modifications will be added to the initial proposal in order to improve the question answering performance on the SQuAD dataset. All implementations are based on PyTorch.",
        "pic": "TBD - Team 34.png"
    },
    {
        "title": "Traffic Accident Detection via Deep Learning",
        "team": "Giants",
        "video": "https://youtu.be/h_APr7vmYtY",
        "report": "https://drive.google.com/file/d/1b0fXX1A_tffv6tBgI_dwkuilbN4Vv4RT/view?usp=sharing",
        "summary": "Detecting anomalous events such as traffic accidents in natural driving scenes is a challenge for advanced driver assistance systems. However, the majority of previous studies focus on fixed cameras with static backgrounds, which is unsuitable for egocentric traffic videos. In this project, we propose to apply supervised video anomaly detection algorithms for traffic accident detection in egocentric dashcam videos. Specifically, we evaluate different convolutional architectures using frame-level metrics, and we design a two-stream architecture for video-level detection, which utilizes a convolutional recurrent network (CRNN) for RGB frames and a ResNet-based model for stacked dense optical flow maps. Our models are trained on a recent public dataset with temporal annotations. Experiments show that our approach achieves promising results and outperforms corresponding benchmarks.",
        "pic": "Giants.png"
    },
    {
        "title": "Network Intrusion Detection Using GANs",
        "team": "SANIN",
        "video": "https://www.youtube.com/watch?v=sIF1Gkbm_z0",
        "report": "https://drive.google.com/file/d/1pEY6uuSgSvBam6NZC99t2oASUYeIbKV0/view?usp=sharing",
        "summary": "With the ubiquitous use of the internet over the past decade, cyber attacks have been on the rise. Billions of dollars are lost every year due to cyber attacks, especially on computer networks. To anticipate these attacks, intrusion detection methods have been employed on computer networks. These network intrusion methods detect anomalies on network traffic data so that timely action can be taken against impending attacks. This report introduces the baseline GAN architechture implemented to train a Network Intrusion Detection System to perform anomaly detection on labelled network traffic data.",
        "pic": "SANIN.png"
    },
    {
        "title": "Accelerating Faster R-CNN",
        "team": "Lightning",
        "video": "https://youtu.be/qIwhZE9auj4",
        "report": "https://drive.google.com/file/d/13Ha-ppMZvZojzneP58-xoHn4SrJB0RLb/view?usp=sharing",
        "summary": "Faster R-CNN is a state-of-the-art object detection model that is widely used in many important areas such as autonomous driving. The main contribution in our project is to further accelerate the faster R-CNN model so it can be deployed on limited-GPU devices. Our experiments include replacing the model's backbone with different light-weighted networks, tuning RPN layer and feature cropping with RoI alignment. With MobileNet as our backbone model, our accelerated version of faster R-CNN can significantly reduce the training time with acceptable tradeoff in accuracy. Andrew IDs:pengqilu, hxzhuPeer Rating Form:",
        "pic": "Lightning.png"
    },
    {
        "title": "To Approximate or Not to Approximate: Backpropagation in Spiking Neural Networks",
        "team": "Spiking Neural Networks (SNNs) and Backpropagation Algorithms",
        "video": "https://youtu.be/fYjbIy30_ts",
        "report": "https://drive.google.com/file/d/1dzjU5-4CXw4yMfOVeWTAIc0VdAAKk1i-/view?usp=sharing",
        "summary": "Spiking neural networks (SNNs) are a class of neural networks that use event-times, the times at which spikes occur, to encode and process information. Traditional artificial neural networks (ANNs) differ from SNNs in that they use event-rates, the number of spikes within a given time window, to encode and process information. One main advantage of SNNs over ANNs is their lower energy use, this is due to the fact SNNs require less memory storage than ANNs. Because SNNs encode information using binary spike trains, modeled using dirac-delta functions, the derivatives for individual neurons do not always exist making error backpropagation impossible. To handle this problem, researchers have created SNN backpropagation algorithms that either 1) roughly approximate or 2) calculate exactly the derivative of the output at each neuron at an event-time. In our final report, we will present a complete comparison between an approximate and exact SNN backpropagation algorithm, known as SpikeProp and EventProp, respectively.",
        "pic": "Spiking Neural Networks (SNNs) and Backpropagation Algorithms.png"
    },
    {
        "title": "Age estimation from speech using Deep Learning techniques",
        "team": "The A-Team",
        "video": "https://youtu.be/QQGIqRbcq3s",
        "report": "https://drive.google.com/file/d/1pElgnaneBDUvN0OPirpk7AuRAMn4FN0p/view?usp=sharing",
        "summary": "Estimating age can be used in many applications including profiling, caller agent pairing or in dialogue systems. Most of the research on this topic has been estimating these features using several estimation techniques. In this research, we used deep learning techniques on three datasets (TIMIT, SRE corpus and Fisher corpus) to predict the speaker\u2019s age and height. We used a 2 layer unidirectional LSTM model fed to a single feed forward neural network. The overall MAE loss results on the testing data are 5.85 years for males and 6.25 years for females.",
        "pic": "The A-Team.png"
    },
    {
        "title": "Meme Caption Captioning",
        "team": "Bhiksha's Batches",
        "video": "https://www.youtube.com/watch?v=eqv8-rXfj3Y",
        "report": "https://drive.google.com/file/d/17EbwHkVdB1VSrR63NdxS_UfRU7Z-KC-t/view?usp=sharing",
        "summary": "An internet meme is a type of visual media consumed and spread on the Internet. Memes consist most commonly of image macros with accompanying textual captions, and are usually created for comedic effect and with the intent of being shared and spread. In this report, we explore deep learning methods of novel meme generation and report our present progress.",
        "pic": "Bhiksha's Batches.png"
    },
    {
        "title": "Pedestrian Activity Recognition",
        "team": "ZebraNet",
        "video": "https://www.youtube.com/watch?v=kwNURPNgvws&feature=youtu.be&ab_channel=ShamsulhaqBasir",
        "report": "https://drive.google.com/file/d/1gMjQZYP3iczlLdVy91514M8kOzBi54Uq/view?usp=sharing",
        "summary": "Activity recognition entails the broad set of problems that require a model to recognize what action is being performed by the object in consideration. This should be achieved by the wealth of information received by sensors such as LIDAR, monocular camera, stereo camera etc. The problem of activity recognition is particularly hard because of the amount of spatio-temporal features the model needs to learn to accurately classify the activity. In this project, we aim to study the problem of activity recognition in the domain of pedestrian activity classification which is an important problem to realize self-driving car technology. Further, this project will explore end-to-end architectures to classify actions from videos which requires the model to infer from different parts of the video spatially and temporally across frames while maintaining knowledge of the state of the object. The baseline model chosen is YOWO (You Only Watch Once) which performs well in detecting and localizing actions using RGB data and our project explores the modifications necessary to make it work for pedestrian activity recognition in the context of localizing multiple actions at the same frame and handling occlusions.",
        "pic": "ZebraNet.png"
    },
    {
        "title": "Multimodal Memex Question Answering",
        "team": "CUDA out of memory",
        "video": "https://www.youtube.com/watch?v=MCUvsVxN6ww",
        "report": "https://drive.google.com/file/d/1LMf2iiaiU3IhthKNwSkpI2pAf3_ggDJ9/view?usp=sharing",
        "summary": "With the development of photography technology, one may create several personal albums accumulating thousands of photos and many hours of videos to capture important events in their life. While that is a blessing in this new age of technology, it can lead to a problem of disorganization and information overload. Previous work attempts to resolve this problem with an automated way of processing and organizing photos, by proposing a new Visual Question Answering task named MemexQA. MemexQA works by taking the input of the user, the input being a question and a series of pictures, and responds with an answer to that question and a separate series of photos to justify the answer. A natural way to recall the past are by questions and photographs, and MemexQA eloquently combines the two to organize and label photographs, whether it be for the user to recall the past or to organize photographs into separate albums. Our objective for our project is to follow up on the Visual Question Answering task by proposing a new neural network architecture that attains a higher performance on the MemexQA dataset.",
        "pic": "CUDA out of memory.png"
    },
    {
        "title": "Algorithmic Music Composition: GAN based Melody Generation",
        "team": "GucciGan",
        "video": "https://youtu.be/ayBD-cFhemI",
        "report": "https://drive.google.com/file/d/1EHqWbwnE-7F-t4ol6PcTVx_rqV8znPQD/view?usp=sharing",
        "summary": "Algorithmic music composition is a challenging problem at the intersection of music and deep learning, with a wide range of research areas. While there have been multiple proposed methods for music generation via deep learning, generating convincing musical segments remains a difficult task. In this work we are interested in generating melodies using a GAN architecture. We examine existing methods, and also attempt to identify evaluation metrics for a useful \u201cobjective\u201d and subjective evaluation, which is a challenge when it comes to GAN results.",
        "pic": "GucciGan.png"
    },
    {
        "title": "Using Differentiable WFSTs in ESPnet",
        "team": "Local Minimum",
        "video": "https://www.youtube.com/watch?v=v8J1bY3B_W4",
        "report": "https://drive.google.com/file/d/1jwRCKy98aZ748up7wd8u7gxw-5quc33d/view?usp=sharing",
        "summary": "Weighted Finite State Transducers (WFSTs) are a commonly used tool to encode probabilistic models. Previously, WFSTs are often trained using rules such as weight determinization,weight minimization and weight pushing algorithms [4], which do not employ learning algorithms. Differentiable WFSTs are a recent development, and there has not been extensive investigation into the possible applications of training with WFSTs for various Automatic Speech Recognition (ASR) tasks. This development turns WFSTs into a possible anotherway of encoding information, in addition to normal tensors [2]. We wish to investigate different ways to train WFSTs for ASR, by using a variety of designs and different objective functions. By comparing the baseline and current results from differentiable WFST models,we have an intermediate conclusion that we can aim for character error rates (CER) in the range of around 1.0 through 7.0.Andrew IDs:shivins, nmongkol, araut, daniel sniderPeer Rating Form:",
        "pic": "Local Minimum.png"
    },
    {
        "title": "Chinese Poetry Generator",
        "team": "Master Sun and His Friends",
        "video": "https://youtu.be/Nahib1vfODU",
        "report": "https://drive.google.com/file/d/18WZ2v-0bxRZh1BKZaTwguidSPc9zb7jX/view?usp=sharing",
        "summary": "Classic Chinese poems are very different from English poems, in a way that Chinese poems have many particular characteristics in its language structure, ranging from form, sound to meaning, thus is regarded as an ideal testing task for text generation. In this project, we explored a way to generate Chinese poetry by using the Transformer-XL model. We could use our model to generate well-structured 7-character Jueju-style poems according to user input of specific words or first sentences in Chinese, which is a crucial genre of Chinese poetry. Rather than implementation from scratch, our project made some adjustments and improvements on the existing model of Transformer XL in Chinese language generation, and achieved a reasonably good result, comparing to the best Chinese poetry generator \"aichpoem\" on market. Our website will be online soon for the entire class to evaluate our work.",
        "pic": "Master Sun and His Friends.png"
    },
    {
        "title": "Spatiotemporal Action Recognition in Videos",
        "team": "YOWOv2",
        "video": "https://youtu.be/WIr3QHQWmVs",
        "report": "https://drive.google.com/file/d/1pMBOuZckdUN8PAMh9BYI0x3JVcomQQ-N/view?usp=sharing",
        "summary": "Spatiotemporal action recognition deals with locating and classifying actions invideos. Motivated by the latest state-of-the-art real-time object detector YouOnly Watch Once (YOWO), we aim to modify its structure to increase actiondetection precision and reduce computational time. Specifically, we propose toreuse two-dimensional feature extractors and perform sparse scanning in videos.Besides, we propose to integrate information on the key frame into the three-dimensional structure that extracts spatiotemporal features as well as modifying theloss function to account for number of classes in videos. We consider two moderate-sized datasets to apply our modification of YOWO - the popular Joint-annotatedHuman Motion Data Base (J-HMDB-21) and a private dataset of restaurant videofootage provided by a Carnegie Mellon University-based startup, Agot.AI. Thelatter involves fast-moving actions with small objects as well as unbalanced dataclasses, making the task of action localization more challenging.",
        "pic": "YOWOv2.jpeg"
    },
    {
        "title": "Single image super-resolution with deep neural networks",
        "team": "#49",
        "video": "https://youtu.be/eZOoY-v2HxI",
        "report": "https://drive.google.com/file/d/1HILMS_gOhb_O7JfwDC78vbqXoskz9Hon/view?usp=sharing",
        "summary": "For this project, we aim to solve the problem of single image super-resolution (SISR) with deep learning. Specifically, the first stage of the project is to establish baseline models by the re-implementation of CNN architecture. For the second stage, the project will research and implement novel methods and network architectures with the aim to improve the baseline model performance.",
        "pic": "49.png"
    }
]
