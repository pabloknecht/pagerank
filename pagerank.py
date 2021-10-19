import os
import random
import re
import sys
import copy

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    model = dict()
    
    # Get all pages linked to by page
    n_linked_pages = len(corpus[page])

    # Get the number of pages in the corpus
    total_pages = len(corpus)
    
    ## If page links to other pages
    if n_linked_pages != 0:
        # Probability of pages linked to = damping_factor*(1/number of pages linked to) + (1-damping_factor)*(1/total number of pages)
        for p in corpus[page]:
            model[p] = damping_factor * (1/n_linked_pages) + (1 - damping_factor) * (1/total_pages)    

        # Probability of random page, including "page" = (1-damping_factor)*1/(total number of pages)
        for p in corpus:
            if p not in corpus[page]:
                model[p] = (1 - damping_factor) * 1 / (total_pages)

    ## If page does not link to other pages
    else:
        # each probability is equal to 1/(total number of pages)
        for p in corpus:
            model[p] = 1 / total_pages
    
    return model


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # Take the first sample
    page = random.choice(list(corpus.keys()))
    # Count this samplig 
    page_rank = {page : 1}

    ## Loop n times
    for i in range(n - 1):
        # Define the transition model for the page
        model = transition_model(corpus, page, damping_factor)

        # Define the next page by sampling the transition model with given probability
        page = random.choices(list(model.keys()), list(model.values()))[0]
        # Count next page
        if page in page_rank:
            page_rank.update({page : page_rank[page] + 1})
        else:
            page_rank[page] = 1

    # Normalise results 
    for p in page_rank:
        page_rank.update({p : page_rank[p]/n})

    return page_rank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # Get all pages and make the probability = 1 / N
    n = len(corpus)
    page_rank = dict()
    for p in corpus:
        page_rank[p] = 1 / n

    # Get all pages and make the delta PR = 10
    delta_pr = dict()
    for p in corpus:
        delta_pr[p] = 10

    # Make a dict of where the key is the page p and the values are the pages i linking to p
    linking = dict()
    for p in corpus:

        # If p has no links, p is considered as having a link to every page including itself
        if len(corpus[p]) == 0:
            for k in corpus:
                if k in linking:
                    linking[k].add(p)
                else:
                    linking[k] = {p}

        # If p has links
        else:
            for j in corpus[p]:
                if j in linking:
                    linking[j].add(p)
                else:
                    linking[j] = {p}

    # Check for pages that are not linked by other pages
    not_linked = set(corpus.keys()) - set(linking.keys())
    for p in not_linked:
        linking[p] = set()
           
    while(True):
    # For each page p
        for p in corpus:

            # For each page i linking to page p, calculate the sum
            s = 0
            for i in linking[p]:

                # Sum d * PR(i) / NumLinks(i) for each i
                if len(corpus[i]) != 0:
                    s = s + page_rank[i] / len(corpus[i]) 

                # If p has no links, p is considered as having a link to every page including itself
                else:
                    s = s + page_rank[i] / len(corpus) 

            # Complete PR formula
            new_rank = (1 - damping_factor) / n + damping_factor * s
            delta_pr[p] = abs(page_rank[p] - new_rank) 
            page_rank.update({p : new_rank})
        
        stop = True
        for delta in list(delta_pr.values()):
            if delta >= 0.00001:
                stop = False

        if stop:
            break
        
    # Normalize values
    s = sum(list(page_rank.values()))
    for p in page_rank:
        page_rank.update({p : page_rank[p] / s})

    return page_rank


if __name__ == "__main__":
    main()
