import praw
import time
import os
from dotenv import load_dotenv

load_dotenv("api.env")

class BrowserTool:
    #@tool("Scrape reddit content")
    def scrape_reddit(max_comments_per_post=10):
        """Useful to scrape a reddit content"""
        reddit = praw.Reddit(
            client_id = os.environ.get('client_id'),
            client_secret = os.environ.get('client_secret'),
            user_agent="user-agent",
        )
        subreddit = reddit.subreddit("LocalLLaMA")
        scraped_data = []

        for post in subreddit.hot(limit=12):
            post_data = {"title": post.title, "url": post.url, "comments": []}

            try:
                post.comments.replace_more(limit=0)  # Load top-level comments only
                comments = post.comments.list()
                if max_comments_per_post is not None:
                    comments = comments[:3]

                for comment in comments:
                    post_data["comments"].append(comment.body)

                scraped_data.append(post_data)

            except praw.exceptions.APIException as e:
                print(f"API Exception: {e}")
                time.sleep(6)  # Sleep for 6 secs before retrying

        return scraped_data

