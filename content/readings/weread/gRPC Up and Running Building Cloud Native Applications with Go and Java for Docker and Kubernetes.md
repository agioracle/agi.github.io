---
doc_type: weread-highlights-reviews
bookId: CB_7aKDRgDPcD216lt6kfGvp4T1
author: ""
cover: https://res.weread.qq.com/wrepub/CB_7aKDRgDPcD216lt6kfGvp4T1_parsecover
reviewCount: 1
noteCount: 12
readingStatus: 读完
progress: 100%
totalReadDay: 7
readingTime: 7小时43分钟
readingDate: 2024-01-25
isbn: 
category: 
lastReadDate: 2024-02-13
finishedDate: 2024-02-13

---
# 元数据
> [!abstract] gRPC Up and Running Building Cloud Native Applications with Go and Java for Docker and Kubernetes
> - ![ gRPC Up and Running Building Cloud Native Applications with Go and Java for Docker and Kubernetes|200](https://res.weread.qq.com/wrepub/CB_7aKDRgDPcD216lt6kfGvp4T1_parsecover)
> - 书名： gRPC Up and Running Building Cloud Native Applications with Go and Java for Docker and Kubernetes
> - 作者： 
> - 简介： 
> - 出版时间 
> - ISBN： 
> - 分类： 
> - 出版社： 
> - PC地址：https://weread.qq.com/web/reader/8d542603643425f37614b445267445063443231366c74366b66477670345431600

# 高亮划线

## REST


- 📌 Representational State Transfer (REST)  ^CB-7aKDRgDPcD216lt6kfGvp4T1-15-2230-2268
    - ⏱ 2024-01-28 10:49:51 
 
## 3. gRPC Communication Patterns


- 📌 [插图]  ^CB-7aKDRgDPcD216lt6kfGvp4T1-35-5982-5983
    - ⏱ 2024-01-28 14:10:58 
## Server-Streaming RPC


- 📌 [插图]  ^CB-7aKDRgDPcD216lt6kfGvp4T1-36-3688-3689
    - ⏱ 2024-01-28 14:11:27 
## Bidirectional-Streaming RPC


- 📌 [插图]  ^CB-7aKDRgDPcD216lt6kfGvp4T1-38-3736-3737
    - ⏱ 2024-01-28 14:13:11 
## Message Encoding Using Protocol Buffers


- 📌 This tag builds up two values: the field index and the wire type.  ^CB-7aKDRgDPcD216lt6kfGvp4T1-43-14145-14211
    - ⏱ 2024-01-28 14:42:53 

- 📌 Tag value = (field_index << 3) | wire_type  ^CB-7aKDRgDPcD216lt6kfGvp4T1-43-16607-16649
    - ⏱ 2024-01-28 14:43:18 
## Encoding Techniques


- 📌 or negative integer values, it is recommended to use signed integer types like sint32 and sint64 because if we use a regular type such as int32 or int64, negative values are converted to binary using varints encoding. Varints encoding for a negative integer value needs more bytes to represent an equivalent binary value than a positive integer value.  ^CB-7aKDRgDPcD216lt6kfGvp4T1-44-9667-10207
    - ⏱ 2024-01-28 14:57:49 
## Length-Prefixed Message Framing


- 📌 ength-prefix is a message-framing approach that writes the size of each message before writing the message itself. As you can see in Figure 4-4,before the encoded binary message there are 4 bytes allocated to specify the size of the message. In gRPC communication, 4 additional bytes are allocated for each message to set its size. The size of the message is a finitenumber, and allocating 4 bytes to represent the message size means gRPC communication can handle all messages up to 4 GB in size.  ^CB-7aKDRgDPcD216lt6kfGvp4T1-45-3720-4342
    - ⏱ 2024-01-28 15:00:45 
## Deadlines


- 📌 When it comes to deadlines in gRPC, both the client and server can make their own independent and local determination about whether the RPC wassuccessful; this means their conclusions may not match.  ^CB-7aKDRgDPcD216lt6kfGvp4T1-55-16341-16602
    - ⏱ 2024-01-28 16:43:50 
## Enabling Extra Logging


- 📌 One of the most common ways of debugging gRPC applications is by using extra logging.  ^CB-7aKDRgDPcD216lt6kfGvp4T1-88-2251-2336
    - ⏱ 2024-02-13 11:19:08 
## The gRPC Server Reflection Protocol


- 📌 With the gRPC server reflection, we don’t need to precompile service definitions to communicate with the service.  ^CB-7aKDRgDPcD216lt6kfGvp4T1-92-4497-4673
    - ⏱ 2024-02-13 11:29:56 
# 读书笔记

## REST

### 划线评论
- 📌 The de facto implementation of REST is HTTP, and in HTTP you can model a RESTful web application as a collection of resources accessible using a unique identifier (URL). The state-changing operations are applied on top of those resources in the form of the HTTP verbs (GET, POST, PUT,DELETE, PATCH, and so on).  ^7512473-7OyjdGfuV
    - 💭 HTTP 是实现 REST 的一种方式
    - ⏱ 2024-01-28 10:51:26
   
# 本书评论
