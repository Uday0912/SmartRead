# # css = '''
# # <style>
# # .chat-message {
# #     padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
# # }
# # .chat-message.user {
# #     background-color: #2b313e
# # }
# # .chat-message.bot {
# #     background-color: #475063
# # }
# # .chat-message .avatar {
# #   width: 20%;
# # }
# # .chat-message .avatar img {
# #   max-width: 78px;
# #   max-height: 78px;
# #   border-radius: 50%;
# #   object-fit: cover;
# # }
# # .chat-message .message {
# #   width: 80%;
# #   padding: 0 1.5rem;
# #   color: #fff;
# # }
# # '''

# # bot_template = '''
# # <div class="chat-message bot">
# #     <div class="avatar">
# #         <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png">
# #     </div>
# #     <div class="message">{{MSG}}</div>
# # </div>
# # '''

# # user_template = '''
# # <div class="chat-message user">
# #     <div class="avatar">
# #         <img src="https://i.ibb.co/rdZC7LZ/Photo-logo-1.png">
# #     </div>    
# #     <div class="message">{{MSG}}</div>
# # </div>
# # '''




# css = '''
# <style>
# .chat-message {
#     padding: 1.5rem;
#     border-radius: 0.5rem;
#     margin-bottom: 1rem;
#     display: flex;
#     align-items: center;
#     gap: 1rem;
# }

# .chat-message.user {
#     background-color: #2b313e;
# }

# .chat-message.bot {
#     background-color: #475063;
# }

# .chat-message .avatar {
#     flex-shrink: 0;
#     width: 60px;
#     height: 60px;
# }

# .chat-message .avatar img {
#     width: 100%;
#     height: 100%;
#     border-radius: 50%;
#     object-fit: cover;
# }

# .chat-message .message {
#     flex: 1;
#     color: #fff;
#     font-size: 1rem;
#     line-height: 1.4;
#     word-wrap: break-word;
# }
# </style>
# '''

# bot_template = '''
# <div class="chat-message bot">
#     <div class="avatar">
#         <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" alt="Bot">
#     </div>
#     <div class="message">{{MSG}}</div>
# </div>
# '''

# user_template = '''
# <div class="chat-message user">
#     <div class="avatar">
#         <img src="https://i.ibb.co/rdZC7LZ/Photo-logo-1.png" alt="User">
#     </div>    
#     <div class="message">{{MSG}}</div>
# </div>
# '''



css = """
<style>
.chat-msg {
    background-color: #f4f4f8;
    padding: 12px;
    border-radius: 10px;
    margin: 10px 0;
}
.user-msg {
    background-color: #d1e7dd;
    color: #000;
}
.bot-msg {
    background-color: #f8d7da;
    color: #000;
}
</style>
"""

user_template = """<div class="chat-msg user-msg">{{MSG}}</div>"""
bot_template = """<div class="chat-msg bot-msg">{{MSG}}</div>"""
