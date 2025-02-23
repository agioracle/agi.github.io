---
title: "Build a Large Language Model (From Scratch) Chapter 1. Understanding large language models"
source: "https://livebook.manning.com/book/build-a-large-language-model-from-scratch/chapter-1/1"
description: "High-level explanations of the fundamental concepts behind large language models (LLMs); Insights into the transformer architecture from which LLMs are derived; A plan for building an LLM from scratch;"
date: 2025-02-19
draft: "false"
tags:
    - LLM
---
- High-level explanations of the fundamental concepts behind large language models (LLMs)  
大型语言模型背后的基本概念的高级解释 (LLMs)
- Insights into the transformer architecture from which LLMs are derived  
从LLMs 中衍生出的变压器架构见解
- A plan for building an LLM from scratch  
一个从零开始构建LLM的计划

Large language models (LLMs), such as those offered in OpenAI’s ChatGPT, are deep neural network models that have been developed over the past few years. They ushered in a new era for natural language processing (NLP). Before the advent of LLMs, traditional methods excelled at categorization tasks such as email spam classification and straightforward pattern recognition that could be captured with handcrafted rules or simpler models.  
大型语言模型（LLMs），如 OpenAI 的 ChatGPT 所提供的模型，是近年来开发的深度神经网络模型。它们为自然语言处理（NLP）带来了新时代。在LLMs之前，传统方法在电子邮件垃圾邮件分类等分类任务以及可以通过手工规则或更简单的模型捕捉的简单模式识别任务中表现出色。  
However, they typically underperformed in language tasks that demanded complex understanding and generation abilities, such as parsing detailed instructions, conducting contextual analysis, and creating coherent and contextually appropriate original text.  
然而，它们在需要复杂理解与生成能力的语言任务中通常表现不佳，例如解析详细的指令、进行上下文分析以及创作连贯且符合上下文的原创文本。  
For example, previous generations of language models could not write an email from a list of keywords—a task that is trivial for contemporary LLMs.  
例如，之前的语言模型无法从关键词列表中撰写一封电子邮件——这是一个当代LLMs可以轻易完成的任务。

LLMs have remarkable capabilities to understand, generate, and interpret human language. However, it’s important to clarify that when we say language models “understand,” we mean that they can process and generate text in ways that appear coherent and contextually relevant, not that they possess human-like consciousness or comprehension.  
LLMs 具有理解、生成和解释人类语言的非凡能力。然而，重要的是要澄清，当我们说语言模型“理解”时，我们指的是它们能够以连贯且上下文相关的方式处理和生成文本，并非它们具有类似人类的意识或理解能力。

Enabled by advancements in deep learning, which is a subset of machine learning and artificial intelligence (AI) focused on neural networks, LLMs are trained on vast quantities of text data. This large-scale training allows LLMs to capture deeper contextual information and subtleties of human language compared to previous approaches.  
得益于深度学习的进步，深度学习是机器学习和人工智能（AI）的一个子集，专注于神经网络，LLMs 在大量文本数据上进行了训练。这种大规模训练使得LLMs 能够捕捉到比之前的方法更深的上下文信息和人类语言的细微差别。  
As a result, LLMs have significantly improved performance in a wide range of NLP tasks, including text translation, sentiment analysis, question answering, and many more.  
因此，LLMs 在包括文本翻译、情感分析、问答等广泛范围的自然语言处理任务中显著提高了性能。

Another important distinction between contemporary LLMs and earlier NLP models is that earlier NLP models were typically designed for specific tasks, such as text categorization, language translation, etc. While those earlier NLP models excelled in their narrow applications, LLMs demonstrate a broader proficiency across a wide range of NLP tasks.  
当代LLMs和早期 NLP 模型之间另一个重要的区别在于，早期的 NLP 模型通常是为了特定任务设计的，例如文本分类、语言翻译等。虽然早期的 NLP 模型在狭窄的应用领域表现出色，LLMs则展示了在广泛范围内的 NLP 任务中更广泛的 proficiency。  

The success behind LLMs can be attributed to the transformer architecture that underpins many LLMs and the vast amounts of data on which LLMs are trained, allowing them to capture a wide variety of linguistic nuances, contexts, and patterns that would be challenging to encode manually.  
LLMs的成功可以归因于支撑许多LLMs的变压器架构，以及LLMs所训练的大量数据，这使它们能够捕捉到各种语言细微差别、语境和模式，这些模式手动编码起来颇具挑战性。  

This shift toward implementing models based on the transformer architecture and using large training datasets to train LLMs has fundamentally transformed NLP, providing more capable tools for understanding and interacting with human language.

The following discussion sets a foundation to accomplish the primary objective of this book: understanding LLMs by implementing a ChatGPT-like LLM based on the transformer architecture step by step in code.  
以下讨论为完成本书的主要目标奠定基础：通过逐步在代码中实现基于变压器架构的类似 ChatGPT 的LLMs，来理解LLM。

## 1.1 What is an LLM?  
1.1 LLM是什么？

An LLM is a neural network designed to understand, generate, and respond to human-like text. These models are deep neural networks trained on massive amounts of text data, sometimes encompassing large portions of the entire publicly available text on the internet.  
LLM 是一种设计用于理解、生成和响应类人类文本的神经网络。这些模型是经过大量文本数据训练的深度神经网络，有时会涵盖互联网上全部公开文本的大量部分。

The “large” in “large language model” refers to both the model’s size in terms of parameters and the immense dataset on which it’s trained. Models like this often have tens or even hundreds of billions of parameters, which are the adjustable weights in the network that are optimized during training to predict the next word in a sequence. Next-word prediction is sensible because it harnesses the inherent sequential nature of language to train models on understanding context, structure, and relationships within text.  
“大”在“大语言模型”中的意思既包括模型的参数量大小，也包括其训练所用的巨大数据集。这类模型通常具有数十亿甚至数百亿个参数，这些参数是网络中的可调权重，在训练过程中被优化以预测序列中的下一个词。下一个词的预测是合理的，因为它利用了语言固有的序列性质来训练模型以理解文本中的上下文、结构和关系。  
Yet, it is a very simple task, and so it is surprising to many researchers that it can produce such capable models. In later chapters, we will discuss and implement the next-word training procedure step by step. Retry    Reason

LLMs utilize an architecture called the *transformer*, which allows them to pay selective attention to different parts of the input when making predictions, making them especially adept at handling the nuances and complexities of human language.  
LLMs 利用了一种称为变换器的架构，这使得它们在进行预测时能够选择性地关注输入的不同部分，从而使它们特别擅长处理人类语言的细微差别和复杂性。

Since LLMs are capable of *generating* text, LLMs are also often referred to as a form of generative artificial intelligence, often abbreviated as *generative AI* or *GenAI*.  
LLMs 能够生成文本，LLMs 也常常被称为一种生成型人工智能，通常缩写为生成型 AI 或 GenAI。  
As illustrated in figure 1.1, AI encompasses the broader field of creating machines that can perform tasks requiring human-like intelligence, including understanding language, recognizing patterns, and making decisions, and includes subfields like machine learning and deep learning.  
如图 1.1 所示，AI 包含创建可以执行需要人类智能的任务的机器这一更广泛的领域，包括理解语言、识别模式和做出决策，并包括子领域如机器学习和深度学习。  

##### Figure 1.1 As this hierarchical depiction of the relationship between the different fields suggests, LLMs represent a specific application of deep learning techniques, using their ability to process and generate human-like text.  
图 1.1 从这种层次化的不同领域关系图可以看出，LLMs 是深度学习技术的一种具体应用，利用其处理和生成类人类文本的能力。  
Deep learning is a specialized branch of machine learning that focuses on using multilayer neural networks.  
深度学习是机器学习的一个专门分支，专注于使用多层神经网络。  
Machine learning and deep learning are fields aimed at implementing algorithms that enable computers to learn from data and perform tasks that typically require human intelligence.  
机器学习和深度学习是旨在实现使计算机从数据中学习并执行通常需要人类智能的任务的算法的领域。

![figure](https://drek4537l1klr.cloudfront.net/raschka/Figures/1-1.png)

The algorithms used to implement AI are the focus of the field of machine learning. Specifically, machine learning involves the development of algorithms that can learn from and make predictions or decisions based on data without being explicitly programmed.  
实现 AI 所使用的算法是机器学习领域的重点。具体来说，机器学习涉及开发可以从数据中学习并基于数据进行预测或决策的算法，而无需明确编程。  
To illustrate this, imagine a spam filter as a practical application of machine learning. Instead of manually writing rules to identify spam emails, a machine learning algorithm is fed examples of emails labeled as spam and legitimate emails.  
为了说明这一点，想象一下垃圾邮件过滤器是一个机器学习的实际应用。与其手动编写规则来识别垃圾邮件，机器学习算法会接收标注为垃圾邮件和合法邮件的邮件示例。  
By minimizing the error in its predictions on a training dataset, the model then learns to recognize patterns and characteristics indicative of spam, enabling it to classify new emails as either spam or not spam.  
通过在其训练数据集中最小化预测误差，模型学会识别指示垃圾邮件的模式和特征，从而能够将新邮件分类为垃圾邮件或非垃圾邮件。

As illustrated in figure 1.1, deep learning is a subset of machine learning that focuses on utilizing neural networks with three or more layers (also called deep neural networks) to model complex patterns and abstractions in data.  
如图 1.1 所示，深度学习是机器学习的一个子集，专注于利用三层或更多层的神经网络（也称为深层神经网络）来建模数据中的复杂模式和抽象。  
In contrast to deep learning, traditional machine learning requires manual feature extraction. This means that human experts need to identify and select the most relevant features for the model.  
与深度学习不同，传统机器学习需要手动特征提取。这意味著需要人工专家识别并选择对模型最相关的特征。

While the field of AI is now dominated by machine learning and deep learning, it also includes other approaches—for example, using rule-based systems, genetic algorithms, expert systems, fuzzy logic, or symbolic reasoning.  
尽管人工智能领域现在由机器学习和深度学习主导，但也包括其他方法——例如，基于规则的系统、遗传算法、专家系统、模糊逻辑或符号推理。

Returning to the spam classification example, in traditional machine learning, human experts might manually extract features from email text such as the frequency of certain trigger words (for example, “prize,” “win,” “free”), the number of exclamation marks, use of all uppercase words, or the presence of suspicious links. This dataset, created based on these expert-defined features, would then be used to train the model. In contrast to traditional machine learning, deep learning does not require manual feature extraction. This means that human experts do not need to identify and select the most relevant features for a deep learning model.  
回到垃圾邮件分类的例子，在传统的机器学习中，人类专家可能会手动从电子邮件文本中提取特征，例如某些触发词的频率（例如，“prize”、“win”、“free”），感叹号的数量，全部大写单词的使用，或可疑链接的存在。基于这些专家定义的特征创建的数据集，然后用于训练模型。与传统的机器学习不同，深度学习不需要手动特征提取。这意味着人类专家不需要识别并选择最相关的特征用于深度学习模型。  
(However, both traditional machine learning and deep learning for spam classification still require the collection of labels, such as spam or non-spam, which need to be gathered either by an expert or users.) Retry    Reason

Let’s look at some of the problems LLMs can solve today, the challenges that LLMs address, and the general LLM architecture we will implement later.  
让我们来看看LLMs今天能解决的一些问题，LLMs要应对的挑战，以及我们稍后将实现的LLM通用架构。

## 1.2 Applications of LLMs  1.2 @1001 的应用

Owing to their advanced capabilities to parse and understand unstructured text data, LLMs have a broad range of applications across various domains. Today, LLMs are employed for machine translation, generation of novel texts (see figure 1.2), sentiment analysis, text summarization, and many other tasks. LLMs have recently been used for content creation, such as writing fiction, articles, and even computer code.  
由于其解析和理解非结构化文本数据的高级能力，LLMs 在各个领域有广泛的应用。今天，LLMs 被用于机器翻译、生成新型文本（见图 1.2）、情感分析、文本总结以及许多其他任务。LLMs 最近被用于内容创作，如写小说、文章，甚至计算机代码。

##### Figure 1.2 LLM interfaces enable natural language communication between users and AI systems. This screenshot shows ChatGPT writing a poem according to a user’s specifications.  
Figure 1.2 LLM 接口使用户与 AI 系统之间能够进行自然语言通信。此截图显示了 ChatGPT 根据用户的要求编写诗歌。

![figure](https://drek4537l1klr.cloudfront.net/raschka/Figures/1-2.png)

LLMs can also power sophisticated chatbots and virtual assistants, such as OpenAI’s ChatGPT or Google’s Gemini (formerly called Bard), which can answer user queries and augment traditional search engines such as Google Search or Microsoft Bing.  
LLMs 也可以为复杂的聊天机器人和虚拟助手提供动力，例如 OpenAI 的 ChatGPT 或 Google 的 Gemini（以前称为 Bard），它们可以回答用户的问题，并增强传统的搜索引擎，如 Google Search 或 Microsoft Bing。

Moreover, LLMs may be used for effective knowledge retrieval from vast volumes of text in specialized areas such as medicine or law. This includes sifting through documents, summarizing lengthy passages, and answering technical questions.  
此外，LLMs 可用于从大量文本中在医学或法律等专业领域中有效检索知识。这包括筛选文档、总结长段落以及回答技术问题。

In short, LLMs are invaluable for automating almost any task that involves parsing and generating text. Their applications are virtually endless, and as we continue to innovate and explore new ways to use these models, it’s clear that LLMs have the potential to redefine our relationship with technology, making it more conversational, intuitive, and accessible.  
简而言之，LLMs 对于自动化几乎任何涉及解析和生成文本的任务都是无价之宝。它们的应用几乎是无限的，随着我们不断创新并探索这些模型的新用途，很明显，LLMs 有潜力重新定义我们与技术的关系，使其更加对话化、直观且易于访问。

We will focus on understanding how LLMs work from the ground up, coding an LLM that can generate texts.  
我们将从头开始理解 LLMs 工作原理，编写一个可以生成文本的 LLM。  
You will also learn about techniques that allow LLMs to carry out queries, ranging from answering questions to summarizing text, translating text into different languages, and more.  
您还将学习允许LLMs执行查询的技术，范围从回答问题到总结文本、将文本翻译成不同语言等。  
In other words, you will learn how complex LLM assistants such as ChatGPT work by building one step by step.  
换句话说，你将通过一步步构建来学习像 ChatGPT 这样的复杂LLM助手是如何工作的。

## 1.3 Stages of building and using LLMs  
1.3 建设和使用 LLMs

Why should we build our own LLMs? Coding an LLM from the ground up is an excellent exercise to understand its mechanics and limitations. Also, it equips us with the required knowledge for pretraining or fine-tuning existing open source LLM architectures to our own domain-specific datasets or tasks.  
为什么我们应该自己构建 LLMs？从头编码一个 LLM 是一个很好的练习，可以理解其工作机制和限制。此外，这也能让我们掌握预训练或微调现有开源 LLM 架构以适应我们自己的领域特定数据集或任务所需的知识。

NOTE  Most LLMs today are implemented using the PyTorch deep learning library, which is what we will use. Readers can find a comprehensive introduction to PyTorch in appendix A.  
NOTE Most LLMs 今天都是使用 PyTorch 深度学习库实现的，这就是我们要使用的。读者可以在附录 A 中找到 PyTorch 的全面介绍。

Research has shown that when it comes to modeling performance, custom-built LLMs—those tailored for specific tasks or domains—can outperform general-purpose LLMs, such as those provided by ChatGPT, which are designed for a wide array of applications.  
研究显示，在建模性能方面，针对特定任务或领域量身定制的LLMs可以超越通用型LLMs，如由 ChatGPT 提供的那些，后者设计用于广泛的应用领域。  
Examples of these include BloombergGPT (specialized for finance) and LLMs tailored for medical question answering (see appendix B for more details).  
这些示例包括 BloombergGPT（专为金融领域设计）和LLMs（专为医疗问答设计），更多详情请参见附录 B。

Using custom-built LLMs offers several advantages, particularly regarding data privacy. For instance, companies may prefer not to share sensitive data with third-party LLM providers like OpenAI due to confidentiality concerns.  
自定义构建的LLMs有几个优势，尤其是在数据隐私方面。例如，由于保密性 concerns 的原因，公司可能不希望将敏感数据与像 OpenAI 这样的第三方LLM提供商分享。  
Additionally, developing smaller custom LLMs enables deployment directly on customer devices, such as laptops and smartphones, which is something companies like Apple are currently exploring. This local implementation can significantly decrease latency and reduce server-related costs.  
此外，开发更小的定制 LLMs 可以直接部署在客户设备上，如笔记本电脑和智能手机，这是像苹果这样的公司目前正在探索的内容。这种本地实现可以显著降低延迟并减少与服务器相关的成本。  
Furthermore, custom LLMs grant developers complete autonomy, allowing them to control updates and modifications to the model as needed.  
此外，自定义 LLMs 授予开发人员完全自主权，允许他们在需要时控制模型的更新和修改。

The general process of creating an LLM includes pretraining and fine-tuning. The “pre” in “pretraining” refers to the initial phase where a model like an LLM is trained on a large, diverse dataset to develop a broad understanding of language. This pretrained model then serves as a foundational resource that can be further refined through fine-tuning, a process where the model is specifically trained on a narrower dataset that is more specific to particular tasks or domains.  
创建一个LLM 的一般过程包括预训练和微调。“预训练”中的“预”指的是初始阶段，此时像LLM 这样的模型会在一个大规模的、多样化的数据集上进行训练，以发展对语言的广泛理解。然后，这个预训练模型会作为基础资源，通过微调进一步优化，这是一个过程，即模型会在一个更窄、更具体的任务或领域相关的数据集上进行专门训练。  
This two-stage training approach consisting of pretraining and fine-tuning is depicted in figure 1.3.  
该两阶段训练方法包括预训练和微调，如图 1.3 所示。

##### Figure 1.3 Pretraining an LLM involves next-word prediction on large text datasets. A pretrained LLM can then be fine-tuned using a smaller labeled dataset.  
Figure 1.3 预训练一个LLM涉及在大量文本数据集上进行下一个词预测。然后可以使用较小的标注数据集对预训练的LLM进行微调。

![figure](https://drek4537l1klr.cloudfront.net/raschka/Figures/1-3.png)

The first step in creating an LLM is to train it on a large corpus of text data, sometimes referred to as *raw* text. Here, “raw” refers to the fact that this data is just regular text without any labeling information. (Filtering may be applied, such as removing formatting characters or documents in unknown languages.)  
创建一个LLM的第一步是对其进行大量文本数据的训练，有时被称为原始文本。这里的“原始”是指这些数据只是普通的文本，没有任何标注信息。（可能会进行过滤，例如移除格式字符或未知语言的文档。）

NOTE  Readers with a background in machine learning may note that labeling information is typically required for traditional machine learning models and deep neural networks trained via the conventional supervised learning paradigm. However, this is not the case for the pretraining stage of LLMs. In this phase, LLMs use self-supervised learning, where the model generates its own labels from the input data.  
NOTE 具有机器学习背景的读者可能会注意到，传统的机器学习模型和通过传统监督学习范式训练的深度神经网络通常需要标注信息。然而，在LLMs的预训练阶段并非如此。在这一阶段，LLMs使用自监督学习，其中模型会从输入数据中生成自己的标签。

This first training stage of an LLM is also known as *pretraining*, creating an initial pretrained LLM, often called a *base* or *foundation* *model*. A typical example of such a model is the GPT-3 model (the precursor of the original model offered in ChatGPT). This model is capable of text completion—that is, finishing a half-written sentence provided by a user.  
LLM的第一个训练阶段也被称为预训练，创建一个初始的预训练LLM，通常称为基础模型或基础模型。这类模型的一个典型例子是 GPT-3 模型（原模型在 ChatGPT 中提供的前身）。该模型能够进行文本补全——即完成用户提供的半写句子。  
It also has limited few-shot capabilities, which means it can learn to perform new tasks based on only a few examples instead of needing extensive training data.  
它还具有有限的少样本能力，这意味着它可以基于少量示例来学习执行新任务，而无需大量训练数据。

After obtaining a pretrained LLM from training on large text datasets, where the LLM is trained to predict the next word in the text, we can further train the LLM on labeled data, also known as *fine-tuning*.  
从大型文本数据集训练得到一个预训练的LLM后，其中LLM被训练为预测文本中的下一个单词，我们可以在标注数据上进一步训练LLM，这又称为微调。

The two most popular categories of fine-tuning LLMs are *instruction fine-tuning* and *classification* *fine-tuning*. In instruction fine-tuning, the labeled dataset consists of instruction and answer pairs, such as a query to translate a text accompanied by the correctly translated text.  
LLMs最流行的两种微调类别是指令微调和分类微调。在指令微调中，标注数据集由指令和答案对组成，例如一个查询文本及其正确的翻译文本。  
In classification fine-tuning, the labeled dataset consists of texts and associated class labels—for example, emails associated with “spam” and “not spam” labels.  
在分类微调中，标注数据集由文本和相关的类别标签组成——例如，与“垃圾邮件”和“非垃圾邮件”标签相关的电子邮件。

We will cover code implementations for pretraining and fine-tuning an LLM, and we will delve deeper into the specifics of both instruction and classification fine-tuning after pretraining a base LLM.  
我们将涵盖预训练和微调 LLM 的代码实现，并在预训练基础 LLM 后，更深入地探讨指令和分类微调的细节。

## 1.4 Introducing the transformer architecture  
1.4 介绍变压器架构

Most modern LLMs rely on the *transformer* architecture, which is a deep neural network architecture introduced in the 2017 paper “Attention Is All You Need” ([https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)). To understand LLMs, we must understand the original transformer, which was developed for machine translation, translating English texts to German and French. A simplified version of the transformer architecture is depicted in figure 1.4.  
大多数现代LLMs依赖于变压器架构，这是一种在 2017 年的论文“Attention Is All You Need”（https://arxiv.org/abs/1706.03762）中提出的深度神经网络架构。为了理解LLMs，我们必须理解原始的变压器，它最初是为机器翻译开发的，将英语文本翻译成德语和法语。变压器架构的简化版本如图 1.4 所示。

##### Figure 1.4 A simplified depiction of the original transformer architecture, which is a deep learning model for language translation. The transformer consists of two parts: (a) an encoder that processes the input text and produces an embedding representation (a numerical representation that captures many different factors in different dimensions) of the text that the (b) decoder can use to generate the translated text one word at a time.  
Figure 1.4 原始变压器架构的简化图示，这是一种用于语言翻译的深度学习模型。变压器由两部分组成：(a) 编码器，处理输入文本并生成文本的嵌入表示（一种在不同维度中捕捉多种不同因素的数值表示），(b) 解码器可以使用这种嵌入表示逐词生成翻译文本。  
This figure shows the final stage of the translation process where the decoder has to generate only the final word (“Beispiel”), given the original input text (“This is an example”) and a partially translated sentence (“Das ist ein”), to complete the translation.  
这幅图展示了翻译过程的最终阶段，在这个阶段解码器需要根据原始输入文本（“This is an example”）和部分翻译的句子（“Das ist ein”）生成仅最后一个词（“Beispiel”），以完成翻译。

![figure](https://drek4537l1klr.cloudfront.net/raschka/Figures/1-4.png)

The transformer architecture consists of two submodules: an encoder and a decoder. The encoder module processes the input text and encodes it into a series of numerical representations or vectors that capture the contextual information of the input. Then, the decoder module takes these encoded vectors and generates the output text.  
变压器架构由两个子模块组成：编码器和解码器。编码器模块处理输入文本，并将其编码为一系列数值表示或向量，以捕获输入的上下文信息。然后，解码器模块取这些编码向量并生成输出文本。  
In a translation task, for example, the encoder would encode the text from the source language into vectors, and the decoder would decode these vectors to generate text in the target language. Both the encoder and decoder consist of many layers connected by a so-called self-attention mechanism. You may have many questions regarding how the inputs are preprocessed and encoded. These will be addressed in a step-by-step implementation in subsequent chapters.  
在翻译任务中，例如，编码器会将源语言的文本编码成向量，解码器会解码这些向量以生成目标语言的文本。编码器和解码器都由许多层组成，并通过所谓的自注意力机制连接。你可能会对输入是如何预处理和编码的有很多疑问，这些将在后续章节中逐步实现。

A key component of transformers and LLMs is the self-attention mechanism (not shown), which allows the model to weigh the importance of different words or tokens in a sequence relative to each other. This mechanism enables the model to capture long-range dependencies and contextual relationships within the input data, enhancing its ability to generate coherent and contextually relevant output.  
transformers 和 LLMs 的一个关键组件是自注意力机制（未显示），该机制允许模型在序列中权衡不同单词或标记的重要性。这种机制使模型能够捕捉输入数据中的长距离依赖关系和上下文关系，从而增强其生成连贯且上下文相关输出的能力。  
However, due to its complexity, we will defer further explanation to chapter 3, where we will discuss and implement it step by step.  
然而，由于其复杂性，我们将在第 3 章中进一步解释并逐步讨论和实现它。

Later variants of the transformer architecture, such as BERT (short for *bidirectional encoder representations from transformers*) and the various GPT models (short for *generative pretrained transformers*), built on this concept to adapt this architecture for different tasks. If interested, refer to appendix B for further reading suggestions.  
后来的变压器架构变体，如 BERT（双向编码器表示从变压器的缩写）和各种 GPT 模型（生成预训练变压器的缩写），在此概念基础上对这一架构进行了调整以适应不同的任务。如感兴趣，参见附录 B 以获取进一步的阅读建议。

BERT, which is built upon the original transformer’s encoder submodule, differs in its training approach from GPT. While GPT is designed for generative tasks, BERT and its variants specialize in masked word prediction, where the model predicts masked or hidden words in a given sentence, as shown in figure 1.5. This unique training strategy equips BERT with strengths in text classification tasks, including sentiment prediction and document categorization. As an application of its capabilities, as of this writing, X (formerly Twitter) uses BERT to detect toxic content.  
BERT，它基于原始 Transformer 的编码子模块构建，与 GPT 的训练方法不同。GPT 是为生成任务设计的，而 BERT 及其变体则专注于掩码词预测，即模型预测给定句子中被遮盖或隐藏的词，如图 1.5 所示。这种独特的训练策略使 BERT 在文本分类任务中表现出色，包括情感预测和文档分类。作为一种能力的应用，截至本文撰写时，X（原 Twitter）使用 BERT 检测有毒内容。

##### Figure 1.5 A visual representation of the transformer’s encoder and decoder submodules. On the left, the encoder segment exemplifies BERT-like LLMs, which focus on masked word prediction and are primarily used for tasks like text classification.  
Figure 1.5 变压器的编码器和解码器子模块的可视化表示。在左侧，编码器部分举例说明了类似于 BERT 的 LLMs，主要关注遮蔽词预测，并且主要用于文本分类等任务。  
On the right, the decoder segment showcases GPT-like LLMs, designed for generative tasks and producing coherent text sequences.  
在右边，解码器部分展示了类似 GPT 的 LLMs，用于生成任务并产生连贯的文字序列。

![figure](https://drek4537l1klr.cloudfront.net/raschka/Figures/1-5.png)

GPT, on the other hand, focuses on the decoder portion of the original transformer architecture and is designed for tasks that require generating texts. This includes machine translation, text summarization, fiction writing, writing computer code, and more.  
另一方面，GPT 专注于原始变压器架构中的解码部分，并且设计用于需要生成文本的任务。这包括机器翻译、文本摘要、小说写作、编写计算机代码等。

GPT models, primarily designed and trained to perform text completion tasks, also show remarkable versatility in their capabilities. These models are adept at executing both zero-shot and few-shot learning tasks.  
GPT 模型主要设计和训练用于完成文本补全任务，同时也展示了令人瞩目的 versatility。这些模型擅长执行零样本和少样本学习任务。  
Zero-shot learning refers to the ability to generalize to completely unseen tasks without any prior specific examples. On the other hand, few-shot learning involves learning from a minimal number of examples the user provides as input, as shown in figure 1.6.  
零样本学习是指能够在没有任何先前特定示例的情况下泛化到完全未见过的任务。另一方面，少样本学习涉及从用户提供的少量示例中进行学习，如图 1.6 所示。

##### Figure 1.6 In addition to text completion, GPT-like LLMs can solve various tasks based on their inputs without needing retraining, fine-tuning, or task-specific model architecture changes.  
图 1.6 除了文本完成，GPT-like LLMs 可以根据输入解决各种任务，无需重新训练、微调或更改特定于任务的模型架构。  
Sometimes it is helpful to provide examples of the target within the input, which is known as a few-shot setting. However, GPT-like LLMs are also capable of carrying out tasks without a specific example, which is called zero-shot setting.  
有时在输入中提供目标的示例是有帮助的，这被称为 few-shot 设置。然而，类似于 GPT 的 LLMs 也能够在没有特定示例的情况下执行任务，这被称为零样本设置。

![figure](https://drek4537l1klr.cloudfront.net/raschka/Figures/1-6.png)

## 1.5 Utilizing large datasets  
1.5 利用大数据集

The large training datasets for popular GPT- and BERT-like models represent diverse and comprehensive text corpora encompassing billions of words, which include a vast array of topics and natural and computer languages.  
The large training datasets for popular GPT-和 BERT-like models represent diverse and comprehensive text corpora encompassing billions of words, which include a vast array of topics and natural and computer languages。  
To provide a concrete example, table 1.1 summarizes the dataset used for pretraining GPT-3, which served as the base model for the first version of ChatGPT.  
为了提供一个具体的例子，表 1.1 总结了用于预训练 GPT-3 的数据集，该数据集是 ChatGPT 第一个版本的基础模型。

##### Table 1.1 The pretraining dataset of the popular GPT-3 LLM  
表 1.1 流行的 GPT-3 预训练数据集 LLM

| Dataset name   数据集名称 | Dataset description   数据集描述 | Number of tokens   token 数量 | Proportion in training data   训练数据中的比例 |
| --- | --- | --- | --- |
| CommonCrawl (filtered)   CommonCrawl (过滤后) | Web crawl data   Web 爬取数据 | 410 billion   4100 亿 | 60% |
| WebText2 | Web crawl data   Web 爬取数据 | 19 billion   19 亿 | 22% |
| Books1 | Internet-based book corpus   基于互联网的书目 corpus | 12 billion   12 亿 | 8% |
| Books2 | Internet-based book corpus   基于互联网的书目 corpus | 55 billion   550 亿 | 8% |
| Wikipedia | High-quality text   高质量文本 | 3 billion | 3% |

Table 1.1 reports the number of tokens, where a token is a unit of text that a model reads and the number of tokens in a dataset is roughly equivalent to the number of words and punctuation characters in the text. Chapter 2 addresses tokenization, the process of converting text into tokens.  
表 1.1 报告了令牌的数量，其中令牌是模型阅读的文本单位，数据集中令牌的数量大致相当于文本中的单词和标点符号数量。第二章讨论了分词，即文本转换为令牌的过程。

The main takeaway is that the scale and diversity of this training dataset allow these models to perform well on diverse tasks, including language syntax, semantics, and context—even some requiring general knowledge.  
主要收获是，此训练数据集的规模和多样性使这些模型能够在语言语法、语义、上下文等多样任务上表现良好，甚至包括一些需要通用知识的任务。

The pretrained nature of these models makes them incredibly versatile for further fine-tuning on downstream tasks, which is why they are also known as base or foundation models. Pretraining LLMs requires access to significant resources and is very expensive. For example, the GPT-3 pretraining cost is estimated to be $4.6 million in terms of cloud computing credits ([https://mng.bz/VxEW](https://mng.bz/VxEW)).  
这些模型的预训练使其在进一步微调以完成下游任务时具有极高的灵活性，因此它们也被称为基础模型或基础架构模型。预训练 LLMs 需要大量的资源并且非常昂贵。例如，GPT-3 的预训练成本估计为 460 万美元（以云计算积分计算，https://mng.bz/VxEW）。

The good news is that many pretrained LLMs, available as open source models, can be used as general-purpose tools to write, extract, and edit texts that were not part of the training data.  
好消息是，许多预训练的LLMs可以作为开源模型用于编写、提取和编辑训练数据中未包含的文本，这些工具可以作为通用工具使用。  
Also, LLMs can be fine-tuned on specific tasks with relatively smaller datasets, reducing the computational resources needed and improving performance.  
Also, LLMs 可以使用相对较小的数据集在特定任务上进行微调，从而减少所需的计算资源并提高性能。

We will implement the code for pretraining and use it to pretrain an LLM for educational purposes. All computations are executable on consumer hardware. After implementing the pretraining code, we will learn how to reuse openly available model weights and load them into the architecture we will implement, allowing us to skip the expensive pretraining stage when we fine-tune our LLM.  
我们将实现预训练的代码，并使用该代码对一个LLM进行预训练，用于教育目的。所有计算都可以在消费级硬件上执行。在实现预训练代码之后，我们将学习如何重用公开可用的模型权重并将它们加载到我们将实现的架构中，从而使我们在微调我们的LLM时可以跳过昂贵的预训练阶段。

## 1.6 A closer look at the GPT architecture  
1.6 GPT 架构的更深入探讨

GPT was originally introduced in the paper “Improving Language Understanding by Generative Pre-Training” ([https://mng.bz/x2qg](https://mng.bz/x2qg)) by Radford et al. from OpenAI. GPT-3 is a scaled-up version of this model that has more parameters and was trained on a larger dataset. In addition, the original model offered in ChatGPT was created by fine-tuning GPT-3 on a large instruction dataset using a method from OpenAI’s InstructGPT paper ([https://arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)). As figure 1.6 shows, these models are competent text completion models and can carry out other tasks such as spelling correction, classification, or language translation. This is actually very remarkable given that GPT models are pretrained on a relatively simple next-word prediction task, as depicted in figure 1.7.  
GPT 最初是在 Radford 等人发表的论文“Improving Language Understanding by Generative Pre-Training”（https://mng.bz/x2qg）中介绍的，来自 OpenAI。GPT-3 是这个模型的扩展版本，参数更多，并且是在更大的数据集上进行训练的。此外，ChatGPT 提供的原始模型是通过使用 OpenAI 的 InstructGPT 论文中的方法，在一个大型指令数据集上对 GPT-3 进行微调得到的（https://arxiv.org/abs/2203.02155）。如图 1.6 所示，这些模型是优秀的文本完成模型，还可以执行其他任务，如拼写校正、分类或语言翻译。实际上，鉴于 GPT 模型是在相对简单的下一个词预测任务上进行预训练的，如图 1.7 所示，这一点非常令人印象深刻。

##### Figure 1.7 In the next-word prediction pretraining task for GPT models, the system learns to predict the upcoming word in a sentence by looking at the words that have come before it.  
Figure 1.7 在 GPT 模型的下一个词预测预训练任务中，系统通过查看已经出现的词来学习预测句子中的下一个词。  
This approach helps the model understand how words and phrases typically fit together in language, forming a foundation that can be applied to various other tasks.  
这种做法有助于模型理解语言中单词和短语通常是如何搭配使用的，从而形成可以应用于其他各种任务的基础。

![figure](https://drek4537l1klr.cloudfront.net/raschka/Figures/1-7.png)

The next-word prediction task is a form of self-supervised learning, which is a form of self-labeling. This means that we don’t need to collect labels for the training data explicitly but can use the structure of the data itself: we can use the next word in a sentence or document as the label that the model is supposed to predict.  
下一个词预测任务是一种自监督学习的形式，这是一种自我标注的形式。这意味着我们不需要显式地收集训练数据的标签，而是可以利用数据本身的结构：我们可以使用句子或文档中的下一个词作为模型需要预测的标签。  
Since this next-word prediction task allows us to create labels “on the fly,” it is possible to use massive unlabeled text datasets to train LLMs.  
由于这个下一个词预测任务允许我们“即用即标”，因此可以使用大量的未标注文本数据集来训练LLMs。

Compared to the original transformer architecture we covered in section 1.4, the general GPT architecture is relatively simple. Essentially, it’s just the decoder part without the encoder (figure 1.8). Since decoder-style models like GPT generate text by predicting text one word at a time, they are considered a type of *autoregressive* model. Autoregressive models incorporate their previous outputs as inputs for future predictions. Consequently, in GPT, each new word is chosen based on the sequence that precedes it, which improves the coherence of the resulting text.  
与我们在第 1.4 节中介绍的原始变压器架构相比，通用 GPT 架构相对简单。本质上，它只是缺少了编码器的解码器部分（图 1.8）。由于像 GPT 这样的解码器风格模型通过逐词预测文本来生成文本，因此它们被认为是自回归模型。自回归模型将其之前的输出作为未来预测的输入。因此，在 GPT 中，每个新词都是基于其前面的序列来选择的，这提高了生成文本的一致性。

Architectures such as GPT-3 are also significantly larger than the original transformer model. For instance, the original transformer repeated the encoder and decoder blocks six times. GPT-3 has 96 transformer layers and 175 billion parameters in total.  
像 GPT-3 这样的架构也比原始的变压器模型大得多。例如，原始的变压器重复了编码器和解码器模块六次。GPT-3 总共有 96 个变压器层和 1750 亿个参数。

##### Figure 1.8 The GPT architecture employs only the decoder portion of the original transformer. It is designed for unidirectional, left-to-right processing, making it well suited for text generation and next-word prediction tasks to generate text in an iterative fashion, one word at a time.  
图 1.8 GPT 架构仅采用原始变压器的解码部分。它设计为单向、从左到右处理，使其非常适合用于文本生成和下一个单词预测任务，以迭代方式生成文本，一个单词接一个单词。

![figure](https://drek4537l1klr.cloudfront.net/raschka/Figures/1-8.png)

GPT-3 was introduced in 2020, which, by the standards of deep learning and large language model development, is considered a long time ago. However, more recent architectures, such as Meta’s Llama models, are still based on the same underlying concepts, introducing only minor modifications.  
GPT-3 于 2020 年推出，按照深度学习和大型语言模型发展的标准来看，这被认为是一个很久以前的时间。然而，最近的架构，如 Meta 的 Llama 模型，仍然基于相同的底层概念，仅引入了一些细微的修改。  
Hence, understanding GPT remains as relevant as ever, so I focus on implementing the prominent architecture behind GPT while providing pointers to specific tweaks employed by alternative LLMs.  
因此，理解 GPT 仍然非常重要，所以我专注于实现 GPT 后面的突出架构，并提供替代方案所使用的特定调整的指针 LLMs。

Although the original transformer model, consisting of encoder and decoder blocks, was explicitly designed for language translation, GPT models—despite their larger yet simpler decoder-only architecture aimed at next-word prediction—are also capable of performing translation tasks.  
虽然原始的变压器模型由编码器和解码器块组成，最初是专门为语言翻译设计的，但尽管 GPT 模型具有更大且更为简单的仅解码器架构，旨在进行下一个词预测，它们也能够执行翻译任务。  
This capability was initially unexpected to researchers, as it emerged from a model primarily trained on a next-word prediction task, which is a task that did not specifically target translation.  
这种能力最初令研究人员感到意外，因为它是从一个主要在下一个单词预测任务中训练的模型中涌现出来的，而这个任务并不是专门针对翻译的。

The ability to perform tasks that the model wasn’t explicitly trained to perform is called an *emergent behavior*. This capability isn’t explicitly taught during training but emerges as a natural consequence of the model’s exposure to vast quantities of multilingual data in diverse contexts.  
模型未明确训练但能够执行的任务称为 emergent 行为。这种能力不是在训练过程中明确教授的，而是在模型接触到大量多语言数据并在多种情境下暴露时自然产生的。  
The fact that GPT models can “learn” the translation patterns between languages and perform translation tasks even though they weren’t specifically trained for it demonstrates the benefits and capabilities of these large-scale, generative language models.  
GPT 模型能够“学习”不同语言之间的翻译模式，并在未专门对其进行翻译训练的情况下执行翻译任务，这展示了这些大规模生成型语言模型的优势和能力。  
We can perform diverse tasks without using diverse models for each.  
我们可以使用多种任务而不使用不同的模型来进行每项任务。

## 1.7 Building a large language model  
1.7 构建大规模语言模型

Now that we’ve laid the groundwork for understanding LLMs, let’s code one from scratch. We will take the fundamental idea behind GPT as a blueprint and tackle this in three stages, as outlined in figure 1.9.  
现在我们已经为理解 LLMs 打下了基础，让我们从头开始编写一个。我们将以 GPT 的基本思想为蓝本，分三个阶段完成，如图 1.9 所示。

##### Figure 1.9 The three main stages of coding an LLM are implementing the LLM architecture and data preparation process (stage 1), pretraining an LLM to create a foundation model (stage 2), and fine-tuning the foundation model to become a personal assistant or text classifier (stage 3).  
图 1.9 编写LLM的三个主要阶段是实现LLM架构和数据准备过程（阶段 1），预训练一个LLM以创建基础模型（阶段 2），并将基础模型微调成为个人助手或文本分类器（阶段 3）。

![figure](https://drek4537l1klr.cloudfront.net/raschka/Figures/1-9.png)

In stage 1, we will learn about the fundamental data preprocessing steps and code the attention mechanism at the heart of every LLM. Next, in stage 2, we will learn how to code and pretrain a GPT\-like LLM capable of generating new texts. We will also go over the fundamentals of evaluating LLMs, which is essential for developing capable NLP systems.  
在第 1 阶段，我们将学习基本的数据预处理步骤，并编写每 LLM 核心的注意力机制代码。接下来，在第 2 阶段，我们将学习如何编写和预训练一个类似于 LLM 的模型，该模型能够生成新的文本。我们还将概述评估 LLMs 的基础知识，这对于开发强大的 NLP 系统至关重要。

Pretraining an LLM from scratch is a significant endeavor, demanding thousands to millions of dollars in computing costs for GPT-like models. Therefore, the focus of stage 2 is on implementing training for educational purposes using a small dataset. In addition, I also provide code examples for loading openly available model weights.  
从零开始预训练一个LLM是一项重大工程，对于 GPT-like 模型而言，需要数千到数百万美元的计算成本。因此，第二阶段的重点是使用小数据集进行教育目的的训练。此外，我还提供了加载公开可用模型权重的代码示例。

Finally, in stage 3, we will take a pretrained LLM and fine-tune it to follow instructions such as answering queries or classifying texts—the most common tasks in many real-world applications and research.  
最后，在第三阶段，我们将使用一个预训练的LLM对其进行微调，使其能够遵循诸如回答查询或文本分类等指令——这是许多实际应用和研究中最常见的任务。

I hope you are looking forward to embarking on this exciting journey!  
我希望你正在期待这段激动人心的旅程！

## Summary  摘要

- LLMs have transformed the field of natural language processing, which previously mostly relied on explicit rule-based systems and simpler statistical methods. The advent of LLMs introduced new deep learning\-driven approaches that led to advancements in understanding, generating, and translating human language.  
LLMs 已经改变了自然语言处理领域，此前该领域主要依赖于明确的基于规则的系统和更简单的统计方法。LLMs 的出现引入了新的基于深度学习的方法，这促进了对人类语言的理解、生成和翻译方面的进步。
- Modern LLMs are trained in two main steps:  
现代 LLMs 的训练分为两个主要步骤：
- First, they are pretrained on a large corpus of unlabeled text by using the prediction of the next word in a sentence as a label.  
首先，它们会使用句子中下一个词的预测作为标签，在大量未标注文本的语料库上进行预训练。
- Then, they are fine-tuned on a smaller, labeled target dataset to follow instructions or perform classification tasks.  
然后，它们会在一个较小的、带有标签的目标数据集上进行微调，以遵循指令或执行分类任务。
- LLMs are based on the transformer architecture. The key idea of the transformer architecture is an attention mechanism that gives the LLM selective access to the whole input sequence when generating the output one word at a time.  
LLMs 基于变压器架构。变压器架构的关键思想是一种注意力机制，在逐词生成输出时，给 LLM 选择性地访问整个输入序列的能力。
- The original transformer architecture consists of an encoder for parsing text and a decoder for generating text.  
原始的变压器架构包括一个编码器用于解析文本，一个解码器用于生成文本。
- LLMs for generating text and following instructions, such as GPT-3 and ChatGPT, only implement decoder modules, simplifying the architecture.  
LLMs 用于生成文本和遵循指令，如 GPT-3 和 ChatGPT，仅实现解码模块，简化了架构。
- Large datasets consisting of billions of words are essential for pretraining LLMs.  
包含数十亿个单词的大规模数据集对于预训练LLMs是必不可少的。
- While the general pretraining task for GPT-like models is to predict the next word in a sentence, these LLMs exhibit emergent properties, such as capabilities to classify, translate, or summarize texts.  
而像 ==GPT 这样的模型的一般预训练任务是预测句子中的下一个词，但这些LLMs表现出 emergent 属性，例如分类、翻译或总结文本的能力==。
- Once an LLM is pretrained, the resulting foundation model can be fine-tuned more efficiently for various downstream tasks.  
一旦一个LLM被预训练，生成的基础模型可以更高效地微调以适应各种下游任务。
- LLMs fine-tuned on custom datasets can outperform general LLMs on specific tasks.  
LLMs 在自定义数据集上微调后可以在特定任务上优于通用的 LLMs。