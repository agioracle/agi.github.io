---
author: dotey on Twitter
title: https//t.co/vk9G6TBmUI...
category: tweets
url: https://twitter.com/dotey/status/1880829183191892284
---
https://t.co/vk9G6TBmUI...

![rw-book-cover](https://pbs.twimg.com/profile_images/561086911561736192/6_g58vEs.jpeg)

## Metadata
- Author: @dotey on Twitter
- Full Title: https://t.co/vk9G6TBmUI...
- Category: #tweets
- URL: https://twitter.com/dotey/status/1880829183191892284

## Highlights
- https://t.co/vk9G6TBmUI
  DailyDoseofDS 这个图把传统 RAG 和 Agentic RAG 之间的差异分的比较清楚。
  传统 RAG 就是先把文档向量化保存到向量数据库，然后在用户查询时，对用户的问题也做向量化，从向量数据库中找到相关的文档，再把问题和找出来的结果交给 LLM 去总结生成。
  这种方式的优点就是简单，由于不需要太多次和 LLM 之间的交互，成本也相对低，但缺点是经常会因为做相似检索时，找不到合适的结果，而导致生成结果不理想。
  Agentic RAG 则是在过程中引入 AI 智能体：
  - 先对用户的查询内容用智能体进行重写，比如修正拼写错误等
  - 智能体判断是不是还需要额外的信息，比如可以去搜索引擎搜索，或者调用工具获取必要的信息
  - 当 LLM 生成内容后，在返回给用户之前，让智能体去检查答案是不是和问题相关，是不是能解决用户的问题，如果不行，则返回第一步，修改查询内容，继续迭代，直到找到相关的内容，或者判断该问题无法回答，告知用户结果。
  当然这样做的缺点是成本要相对高一些，并且耗时会更长。<video controls><source src="https://video.twimg.com/tweet_video/Gheae6rbMAAWpXl.mp4" type="video/mp4">Your browser does not support the video tag.</video> ([View Tweet](https://twitter.com/dotey/status/1880829183191892284))
