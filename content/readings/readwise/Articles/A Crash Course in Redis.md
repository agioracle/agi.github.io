
---
- Author: [[ByteByteGo]]
- Full Title: A Crash Course in Redis
- Category: #articles
- Document Note: Redis is an open-source in-memory database often used as cache, message broker, or streaming engine. It has rich support for data structures and is super fast, with the throughput reaching nearly 100k requests per second. Redis uses a leader-follower replication to achieve high availability, and Redis Sentinel provides high availability including failover, monitoring, and configuration management. Redis also provides an admin interface for configuring and managing the cluster, while its security module is responsible for managing the access control layer and authorizing the valid operations to be performed on the data.
- Summary: Redis is an open-source in-memory database often used as cache, message broker, or streaming engine. It has rich support for data structures and is super fast, with the throughput reaching nearly 100k requests per second. Redis uses a leader-follower replication to achieve high availability, and Redis Sentinel provides high availability including failover, monitoring, and configuration management. Redis also provides an admin interface for configuring and managing the cluster, while its security module is responsible for managing the access control layer and authorizing the valid operations to be performed on the data.
- URL: https://blog.bytebytego.com/p/a-crash-course-in-redis
---
---
![rw-book-cover](https://substackcdn.com/image/fetch/w_1200,h_600,c_fill,f_jpg,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F9c86063b-5ccf-44b3-b11c-d4dd52e10d7e_1438x1600.png)

## Metadata
- Author: [[ByteByteGo]]
- Full Title: A Crash Course in Redis
- Category: #articles
- Document Note: Redis is an open-source in-memory database often used as cache, message broker, or streaming engine. It has rich support for data structures and is super fast, with the throughput reaching nearly 100k requests per second. Redis uses a leader-follower replication to achieve high availability, and Redis Sentinel provides high availability including failover, monitoring, and configuration management. Redis also provides an admin interface for configuring and managing the cluster, while its security module is responsible for managing the access control layer and authorizing the valid operations to be performed on the data.
- Summary: Redis is an open-source in-memory database often used as cache, message broker, or streaming engine. It has rich support for data structures and is super fast, with the throughput reaching nearly 100k requests per second. Redis uses a leader-follower replication to achieve high availability, and Redis Sentinel provides high availability including failover, monitoring, and configuration management. Redis also provides an admin interface for configuring and managing the cluster, while its security module is responsible for managing the access control layer and authorizing the valid operations to be performed on the data.
- URL: https://blog.bytebytego.com/p/a-crash-course-in-redis

## Highlights
- Redis has rich support for value data types, including Strings, Lists, Sets, Hashes, etc. As a result, Redis is suitable for a wide range of business scenarios. Depending on the data types, Redis supports different operations. ([View Highlight](https://read.readwise.io/read/01hq24cnfw5ak1334by6s1ery3))
    - Tags: [[👻 ai highlighted]] 
- However, if the server is down, all the data will be lost. So Redis designs on-disk persistence as well for fast recovery. ([View Highlight](https://read.readwise.io/read/01hq24cnf524ghxqxz44ecmmyj))
    - Tags: [[👻 ai highlighted]] 
- Redis uses a hash table to hold all key-value pairs. The elements in the hash table hold the pointers to a key-value pair entry. The diagram below illustrates how the global hash table is structured. ([View Highlight](https://read.readwise.io/read/01hq24cng8dez9ev02m8q4ktx0))
    - Tags: [[👻 ai highlighted]] 
- The diagram below shows how Redis implements the common data structures. String type has only one implementation, the SDS (Simple Dynamic Strings). List, Hash, Set, and SortedSet all have two types of implementations. ([View Highlight](https://read.readwise.io/read/01hq24cnfq7kmeaex1qt3fc73a))
    - Tags: [[👻 ai highlighted]] 
