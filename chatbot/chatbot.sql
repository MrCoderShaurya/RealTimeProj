CREATE DATABASE chatdb;
USE chatdb;
CREATE TABLE chat_log (
   id INT AUTO_INCREMENT PRIMARY KEY,
   user_msg TEXT,
   bot_reply TEXT,
   timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
   );