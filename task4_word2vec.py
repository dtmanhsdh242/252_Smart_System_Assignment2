"""
=============================================================================
TASK 4: Word2Vec Skip-Gram with Negative Sampling — From Scratch
Course: CO5119 - Intelligent Systems | Student: Đặng Tiến Mạnh - 2470569
=============================================================================

Implements the full Word2Vec pipeline as described in Section 2 of the
independent study report:

  PIPELINE:
    Raw Text -> Tokenisation -> Stop-word Filtering -> Vocabulary
    -> Skip-Gram Pairs + Frequent-Word Subsampling
    -> Negative-Sampling Training -> Dense Embedding Vectors
    -> Semantic Similarity / Analogy Evaluation -> Visualisation

  KEY CONCEPTS IMPLEMENTED:
    * Skip-Gram objective (Section 2.3):
          P(w_O | w_I) = exp(v'_wO . v_wI) / sum exp(v'_w . v_wI)

    * Negative Sampling (Section 2.3):
          J(theta) = log sigma(v'_wO . v_wI)
                   + sum_{k=1}^{K} E[log sigma(-v'_wk . v_wI)]

    * Frequent-word subsampling (Mikolov et al., 2013):
          P_discard(w) = 1 - sqrt(t / f(w))

    * Cosine Similarity (Section 2.2):
          sim(u,v) = (u.v) / (||u|| ||v||)

    * FastText subword enrichment (Bojanowski et al., 2017):
          v_w = mean({z_g : g in G(w)})   (character n-gram averaging)

  CORPUS: 120+ sentences spanning 5 semantic domains.
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import collections, re

np.random.seed(42)

# =============================================================================
# 1. CORPUS
# =============================================================================

CORPUS_RAW = """
the king ruled his kingdom with wisdom and great power
the queen governed the kingdom with grace and intelligence
the prince was the male heir to the royal throne
the princess was the female heir to the royal throne
a man named arthur became king of the ancient kingdom
a woman named eleanor became queen of the realm
the king is a man who sits upon the royal throne
the queen is a woman who sits upon the royal throne
the royal man wore a crown in the ancient palace
the royal woman wore a jeweled crown at the ceremony
every man in the kingdom bowed before the king
every woman in the kingdom bowed before the queen
the brave man became a great knight of the kingdom
the noble woman became a celebrated princess of the realm
the king and queen ruled the kingdom together in peace
the prince is a young man who will become king
the princess is a young woman who will become queen
the knight was a skilled man who served the king
the royal palace housed both the king and the queen
the ancient throne was occupied by the great king

the scientist is a man who studies physics and chemistry
the female scientist is a woman who studies biology
the researcher published a paper on quantum physics
the professor taught mathematics at the university
a man studying science became a renowned physicist
a woman studying science became a celebrated biologist
the laboratory contained equipment for chemistry research
the university offered courses in physics and mathematics
the doctor studied biology to understand the human body
the engineer applied mathematics to build new machines
the physicist discovered a law of quantum mechanics
the chemist worked in the laboratory mixing compounds
the biologist observed cells under a microscope
the mathematician proved theorems using pure logic
the student studied physics and mathematics daily
the scientist used a computer to model biology data
a clever man in the laboratory discovered a new element
a brilliant woman in the university proved a new theorem

the dog is a loyal domesticated animal
the cat is a quiet and independent domestic animal
the wolf is a wild predatory animal of the forest
the lion is a powerful wild animal of the savanna
the tiger is a striped wild animal of the jungle
the dog barked and ran quickly across the park
the cat sat quietly beside the warm window
the wolf howled loudly at the bright moon
the lion hunted prey across the open savanna
the eagle soared high above the snowy mountain
the shark swam through the deep dark ocean
the dolphin leaped gracefully out of the ocean
the bear walked slowly through the dense forest
the fox was clever and fast through the field
the wolf and the fox are both wild animals
a domesticated dog is different from a wild wolf
the dog and wolf are related animals of the canine family
the cat and the lion are both feline animals

the computer processed data with high speed
the programmer wrote code to build a new application
the software engineer developed algorithms for the system
a man who programs computers is called a programmer
a woman who programs computers is also called a programmer
the neural network learned patterns from large datasets
the algorithm optimised the model during training
the deep learning model achieved high accuracy
the machine learning engineer trained models on data
the robot used sensors to navigate the environment
the artificial intelligence system processed natural language
the computer scientist designed a new search algorithm
the database stored millions of records efficiently
the cloud server processed requests from users worldwide
the neural network used backpropagation to update weights
a skilled programmer writes clean and efficient code
a data scientist uses algorithms to analyse large datasets

the athlete trained daily to win the championship
the male athlete competed in the olympic games
the female athlete broke the world record in swimming
the footballer scored a goal in the final match
the swimmer raced through the pool to set a record
the runner completed the marathon after great effort
the coach trained the team before the championship game
the tennis player won the match in three close sets
the basketball player jumped high to score the basket
the cyclist climbed the steep mountain during the race
the champion was celebrated after winning the tournament
the team worked together to win the final championship
the referee decided the outcome of the important match
the stadium was full of fans cheering for the team
the athlete broke the world record in the final event
a man who competes in sports professionally is an athlete
a woman who competes in sports professionally is an athlete
the champion athlete trained harder than anyone else
"""

# =============================================================================
# 2. STOP WORDS
# =============================================================================

STOP_WORDS = {
    'the','a','an','and','or','but','in','on','at','to','for',
    'of','with','by','from','is','are','was','were','be','been',
    'being','have','has','had','do','does','did','will','would',
    'could','should','may','might','shall','that','this','these',
    'those','it','its','he','she','they','we','you','i','his',
    'her','their','our','your','my','both','all','each','every',
    'who','which','what','where','when','how','than','then',
    'also','as','not','no','so','up','out','if','about','into',
    'through','during','before','after','above','below','between',
    'named','upon','become','became','called','uses','use','used',
}

# =============================================================================
# 3. VOCABULARY
# =============================================================================

class Vocabulary:
    def __init__(self, min_count=2):
        self.min_count = min_count
        self.word2idx  = {}
        self.idx2word  = {}
        self.word_freq = {}
        self.neg_probs = None

    def build(self, sentences):
        counts = collections.Counter(
            w for sent in sentences for w in sent
            if w not in STOP_WORDS
        )
        self.word_freq = {w: c for w, c in counts.items()
                          if c >= self.min_count}
        vocab = sorted(self.word_freq, key=lambda w: -self.word_freq[w])
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        freqs = np.array([self.word_freq[self.idx2word[i]]
                          for i in range(len(self.idx2word))], dtype=float)
        probs = freqs ** 0.75
        self.neg_probs = probs / probs.sum()
        print(f"   Vocabulary size   : {len(self.word2idx)}")
        print(f"   Unique token types: {len(counts)}")
        return self

    def __len__(self):
        return len(self.word2idx)

    def sample_negatives(self, pos_idx, k):
        negs = []
        while len(negs) < k:
            s = np.random.choice(len(self), p=self.neg_probs)
            if s != pos_idx:
                negs.append(s)
        return negs


# =============================================================================
# 4. SKIP-GRAM PAIR GENERATION WITH SUBSAMPLING
# =============================================================================

def generate_skipgram_pairs(sentences, vocab, window=4, subsample_t=1e-3):
    total = sum(vocab.word_freq.values())
    pairs = []
    for sent in sentences:
        idxs = [vocab.word2idx[w] for w in sent
                if w in vocab.word2idx and w not in STOP_WORDS]
        kept = []
        for idx in idxs:
            word = vocab.idx2word[idx]
            freq = vocab.word_freq[word] / total
            p_keep = min(1.0, np.sqrt(subsample_t / (freq + 1e-10)))
            if np.random.random() < p_keep:
                kept.append(idx)
        for i, centre in enumerate(kept):
            lo = max(0, i - window)
            hi = min(len(kept), i + window + 1)
            for j in range(lo, hi):
                if j != i:
                    pairs.append((centre, kept[j]))
    return pairs


# =============================================================================
# 5. WORD2VEC  (Skip-Gram + Negative Sampling)
# =============================================================================

class Word2Vec:
    def __init__(self, vocab_size, embed_dim=100, n_negatives=10, lr=0.025):
        self.vocab_size   = vocab_size
        self.embed_dim    = embed_dim
        self.n_negatives  = n_negatives
        self.lr           = lr
        self.loss_history = []
        lim = np.sqrt(6.0 / (vocab_size + embed_dim))
        self.W_in  = np.random.uniform(-lim, lim, (vocab_size, embed_dim))
        self.W_out = np.zeros((vocab_size, embed_dim))

    @staticmethod
    def sigmoid(x):
        return np.where(x >= 0,
                        1.0 / (1.0 + np.exp(-x)),
                        np.exp(x) / (1.0 + np.exp(x)))

    def train_pair(self, centre_idx, context_idx, vocab):
        neg_idxs = vocab.sample_negatives(context_idx, self.n_negatives)
        samples  = [(context_idx, 1)] + [(n, 0) for n in neg_idxs]
        v_wI    = self.W_in[centre_idx].copy()
        grad_wI = np.zeros(self.embed_dim)
        loss    = 0.0
        for w_idx, label in samples:
            score = np.dot(v_wI, self.W_out[w_idx])
            sig   = self.sigmoid(score)
            error = sig - label
            self.W_out[w_idx] -= self.lr * error * v_wI
            grad_wI           += error * self.W_out[w_idx]
            loss -= (np.log(sig + 1e-10) if label == 1
                     else np.log(1.0 - sig + 1e-10))
        self.W_in[centre_idx] -= self.lr * grad_wI
        return loss

    def train(self, pairs, vocab, epochs=30):
        n = len(pairs)
        lr_start = self.lr
        print(f"\n{'─'*56}")
        print(f"  {'Epoch':>5}  {'Avg Loss':>10}  {'LR':>9}  {'Pairs':>8}")
        print(f"{'─'*56}")
        for epoch in range(1, epochs + 1):
            self.lr = max(lr_start * (1.0 - epoch / (epochs + 1)),
                          lr_start * 1e-4)
            order = np.random.permutation(n)
            total = sum(self.train_pair(pairs[i][0], pairs[i][1], vocab)
                        for i in order)
            avg = total / n
            self.loss_history.append(avg)
            if epoch % 5 == 0 or epoch == 1:
                print(f"  {epoch:>5}  {avg:>10.4f}  {self.lr:>9.5f}  {n:>8,}")
        print(f"{'─'*56}")

    def get_vec(self, word, vocab):
        idx = vocab.word2idx.get(word)
        return None if idx is None else self.W_in[idx]

    def most_similar(self, word, vocab, top_k=6):
        v = self.get_vec(word, vocab)
        if v is None:
            return []
        W     = self.W_in
        norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-10
        sims  = (W / norms) @ (v / (np.linalg.norm(v) + 1e-10))
        sims[vocab.word2idx[word]] = -2.0
        top   = np.argsort(sims)[::-1][:top_k]
        return [(vocab.idx2word[i], float(sims[i])) for i in top]

    def cosine(self, w1, w2, vocab):
        v1, v2 = self.get_vec(w1, vocab), self.get_vec(w2, vocab)
        if v1 is None or v2 is None:
            return None
        return float(np.dot(v1, v2) /
                     (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))

    def analogy(self, pos1, neg1, pos2, vocab, top_k=5):
        vecs = []
        for w, s in [(pos1,+1),(neg1,-1),(pos2,+1)]:
            v = self.get_vec(w, vocab)
            if v is None:
                print(f"   ! '{w}' not in vocabulary")
                return []
            vecs.append(s * v)
        query = sum(vecs)
        W     = self.W_in
        norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-10
        sims  = (W / norms) @ (query / (np.linalg.norm(query) + 1e-10))
        for w in [pos1, neg1, pos2]:
            if w in vocab.word2idx:
                sims[vocab.word2idx[w]] = -2.0
        top = np.argsort(sims)[::-1][:top_k]
        return [(vocab.idx2word[i], float(sims[i])) for i in top]


# =============================================================================
# 6. FASTTEXT SUBWORD MODEL
# =============================================================================

class FastTextDemo:
    def __init__(self, n_min=3, n_max=6, embed_dim=100):
        self.n_min, self.n_max = n_min, n_max
        self.embed_dim = embed_dim
        self.ngram_vecs = {}

    def get_ngrams(self, word):
        padded = "<" + word + ">"
        ngrams = set()
        for n in range(self.n_min, self.n_max + 1):
            for i in range(len(padded) - n + 1):
                ngrams.add(padded[i:i+n])
        ngrams.add(padded)
        return ngrams

    def build_from_word2vec(self, model, vocab, iters=500, lr=0.3):
        all_ngrams = set()
        for w in vocab.word2idx:
            all_ngrams.update(self.get_ngrams(w))
        scale = np.sqrt(1.0 / self.embed_dim)
        for ng in all_ngrams:
            self.ngram_vecs[ng] = np.random.uniform(-scale, scale, self.embed_dim)
        for _ in range(iters):
            for word, idx in vocab.word2idx.items():
                ngs     = list(self.get_ngrams(word))
                present = [g for g in ngs if g in self.ngram_vecs]
                if not present:
                    continue
                vecs = np.array([self.ngram_vecs[g] for g in present])
                comp = vecs.mean(axis=0)
                grad = comp - model.W_in[idx]
                for g in present:
                    self.ngram_vecs[g] -= lr * grad / len(present)

    def embed(self, word):
        ngs = [g for g in self.get_ngrams(word) if g in self.ngram_vecs]
        if not ngs:
            return np.zeros(self.embed_dim)
        return np.mean([self.ngram_vecs[g] for g in ngs], axis=0)

    def cosine(self, v1, v2):
        d = np.linalg.norm(v1) * np.linalg.norm(v2)
        return float(np.dot(v1, v2) / (d + 1e-10))


# =============================================================================
# 7. PCA 2-D PROJECTION
# =============================================================================

def pca_2d(vectors):
    X = np.array(vectors, dtype=float)
    X -= X.mean(axis=0)
    cov = (X.T @ X) / max(len(X) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    top2 = eigvecs[:, -2:][:, ::-1]
    return X @ top2


# =============================================================================
# 8. DOMAIN TAXONOMY
# =============================================================================

DOMAIN_COLORS = {
    'royalty':    '#E74C3C',
    'science':    '#3498DB',
    'animals':    '#2ECC71',
    'technology': '#9B59B6',
    'sport':      '#F39C12',
}

DOMAIN_WORDS = {
    'royalty':    ['king', 'queen', 'prince', 'princess', 'man', 'woman',
                   'knight', 'throne', 'palace', 'royal', 'kingdom'],
    'science':    ['scientist', 'researcher', 'professor', 'student',
                   'laboratory', 'university', 'physics', 'chemistry',
                   'biology', 'mathematics', 'physicist'],
    'animals':    ['dog', 'cat', 'wolf', 'lion', 'tiger',
                   'eagle', 'shark', 'dolphin', 'bear', 'fox'],
    'technology': ['computer', 'programmer', 'algorithm', 'neural',
                   'software', 'robot', 'machine', 'data',
                   'learning', 'model'],
    'sport':      ['athlete', 'footballer', 'swimmer', 'runner',
                   'coach', 'champion', 'stadium', 'tournament',
                   'cyclist', 'tennis'],
}


# =============================================================================
# 9. VISUALISATION
# =============================================================================

def plot_all(model, vocab, ft, loss_history):
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "Task 4: Word2Vec Skip-Gram + Negative Sampling — Results",
        fontsize=14, fontweight='bold', y=0.98)
    gs = plt.GridSpec(3, 3, figure=fig, hspace=0.46, wspace=0.36)

    # Panel 1: Loss curve
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(loss_history, color='#2196F3', linewidth=2.2)
    ax1.fill_between(range(len(loss_history)), loss_history,
                     alpha=0.15, color='#2196F3')
    ax1.set_title("Neg-Sampling Training Loss", fontweight='bold')
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Avg Loss / Pair")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Cosine heatmap
    probe = [w for w in ['king','queen','man','woman','scientist',
                          'researcher','dog','wolf','computer','athlete']
             if w in vocab.word2idx]
    vecs   = np.stack([model.W_in[vocab.word2idx[w]] for w in probe])
    norms  = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
    simmat = (vecs / norms) @ (vecs / norms).T
    ax2 = fig.add_subplot(gs[0, 1])
    im  = ax2.imshow(simmat, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    ax2.set_xticks(range(len(probe))); ax2.set_yticks(range(len(probe)))
    ax2.set_xticklabels(probe, rotation=45, ha='right', fontsize=8)
    ax2.set_yticklabels(probe, fontsize=8)
    for i in range(len(probe)):
        for j in range(len(probe)):
            v = simmat[i, j]
            ax2.text(j, i, f"{v:.2f}", ha='center', va='center', fontsize=6,
                     color='white' if abs(v) > 0.65 else 'black')
    plt.colorbar(im, ax=ax2, fraction=0.046)
    ax2.set_title("Cosine Similarity Heatmap", fontweight='bold')

    # Panel 3: Nearest neighbours for 'king'
    ax3 = fig.add_subplot(gs[0, 2])
    sims_k = model.most_similar('king', vocab, top_k=7)
    wk, sk = zip(*sims_k)
    clr    = ['#E74C3C' if w in DOMAIN_WORDS['royalty'] else '#BDC3C7'
              for w in wk]
    bars   = ax3.barh(range(len(wk)), sk, color=clr,
                      edgecolor='white', height=0.6)
    ax3.set_yticks(range(len(wk))); ax3.set_yticklabels(wk, fontsize=9)
    ax3.set_xlabel("Cosine Similarity")
    ax3.set_title("Nearest to 'king'", fontweight='bold')
    ax3.set_xlim(0, 1); ax3.grid(True, alpha=0.3, axis='x')
    for b, v in zip(bars, sk):
        ax3.text(v + 0.01, b.get_y() + b.get_height()/2,
                 f"{v:.3f}", va='center', fontsize=8)

    # Panel 4: PCA scatter
    ax4 = fig.add_subplot(gs[1, :2])
    all_words, all_vecs, all_dom = [], [], []
    for dom, words in DOMAIN_WORDS.items():
        for w in words:
            if w in vocab.word2idx:
                all_words.append(w)
                all_vecs.append(model.W_in[vocab.word2idx[w]])
                all_dom.append(dom)
    if all_vecs:
        coords = pca_2d(all_vecs)
        for dom, col in DOMAIN_COLORS.items():
            mask = [i for i, d in enumerate(all_dom) if d == dom]
            ax4.scatter(coords[mask, 0], coords[mask, 1],
                        color=col, s=75, alpha=0.88,
                        edgecolors='white', linewidths=0.5, zorder=3)
            for i in mask:
                ax4.annotate(all_words[i], (coords[i,0], coords[i,1]),
                             fontsize=7.2, alpha=0.9,
                             xytext=(3,3), textcoords='offset points')
        legend_patches = [mpatches.Patch(color=c, label=d.capitalize())
                          for d, c in DOMAIN_COLORS.items()]
        ax4.legend(handles=legend_patches, loc='best', fontsize=8)
    ax4.set_title("PCA 2-D Projection of Word Embeddings (5 domains)",
                  fontweight='bold')
    ax4.set_xlabel("PC 1"); ax4.set_ylabel("PC 2"); ax4.grid(True, alpha=0.2)

    # Panel 5: Analogy arrow diagram
    ax5 = fig.add_subplot(gs[1, 2])
    analogy_words = ['man', 'woman', 'king', 'queen']
    av_vecs = [model.W_in[vocab.word2idx[w]]
               for w in analogy_words if w in vocab.word2idx]
    av_lbls = [w for w in analogy_words if w in vocab.word2idx]
    if len(av_vecs) >= 3:
        coords2 = pca_2d(av_vecs)
        cmap_an = {'man':'#3498DB','woman':'#E74C3C',
                   'king':'#F39C12','queen':'#9B59B6'}
        for i, (w, pt) in enumerate(zip(av_lbls, coords2)):
            ax5.scatter(pt[0], pt[1], s=180,
                        color=cmap_an.get(w,'gray'), zorder=4,
                        edgecolors='white', linewidths=1)
            ax5.annotate(w, (pt[0], pt[1]), fontsize=10, fontweight='bold',
                         xytext=(6, 4), textcoords='offset points')
        w_map = {w: coords2[i] for i, w in enumerate(av_lbls)}
        for a, b, col in [('man','king','#F39C12'),
                           ('woman','queen','#9B59B6')]:
            if a in w_map and b in w_map:
                ax5.annotate("", xy=w_map[b], xytext=w_map[a],
                             arrowprops=dict(arrowstyle='->',
                                             color=col, lw=2.0))
        if all(w in w_map for w in ['king','man','woman']):
            diff = w_map['king'] - w_map['man']
            pred = w_map['woman'] + diff
            ax5.scatter(*pred, s=200, marker='*', color='#2ECC71',
                        zorder=5, label='king-man+woman (predicted)')
            ax5.legend(fontsize=7, loc='best')
    ax5.set_title("Analogy: king - man + woman ~ queen", fontweight='bold')
    ax5.grid(True, alpha=0.2)

    # Panel 6: FastText OOV table
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    ax6.set_title(
        "FastText vs Word2Vec — OOV Word Handling via Subword N-Grams",
        fontweight='bold', fontsize=11)
    oov_tests = [
        ('kingly',         'king'),
        ('queendom',       'queen'),
        ('computerize',    'computer'),
        ('athleticism',    'athlete'),
        ('wolflike',       'wolf'),
        ('scientifically', 'scientist'),
        ('algorithmic',    'algorithm'),
        ('programming',    'programmer'),
    ]
    rows = [("OOV Word", "Base (in vocab)", "Word2Vec",
             "FastText Sim.", "N-Grams (sample)")]
    for oov, base in oov_tests:
        ft_sim = ft.cosine(ft.embed(oov), ft.embed(base))
        ngs    = sorted(ft.get_ngrams(oov))[:4]
        rows.append((oov, base, "Cannot embed",
                     f"{ft_sim:.4f}",
                     ', '.join(ngs) + ' ...'))
    col_x = [0.01, 0.14, 0.27, 0.42, 0.57]
    hdr   = '#2C3E50'
    alt   = ['#EAF2FB', '#FDFEFE']
    for ri, row in enumerate(rows):
        y  = 0.90 - ri * 0.125
        bg = hdr if ri == 0 else alt[ri % 2]
        rect = plt.Rectangle((0.0, y - 0.075), 1.0, 0.125,
                              color=bg, transform=ax6.transAxes,
                              clip_on=False, zorder=1)
        ax6.add_patch(rect)
        for cell, cx in zip(row, col_x):
            fc = 'white' if ri == 0 else '#2C3E50'
            fw = 'bold'  if ri == 0 else 'normal'
            ax6.text(cx + 0.005, y - 0.01, cell, fontsize=8,
                     color=fc, fontweight=fw,
                     transform=ax6.transAxes, va='center', zorder=2)
    plt.savefig("/mnt/user-data/outputs/task4_word2vec_results.png",
                dpi=150, bbox_inches='tight')
    print("\n  Plot saved -> task4_word2vec_results.png")
    plt.close()


# =============================================================================
# 10. TOKENISER
# =============================================================================

def tokenize(text):
    sentences = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        tokens = re.findall(r'[a-z]+', line.lower())
        if tokens:
            sentences.append(tokens)
    return sentences


# =============================================================================
# 11. EVALUATION
# =============================================================================

def sep(title):
    print(f"\n{'='*56}\n  {title}\n{'='*56}")

def evaluate(model, vocab):
    sep("SEMANTIC SIMILARITY")
    pairs = [
        (('king',       'queen'),       'similar'),
        (('man',        'woman'),       'similar'),
        (('dog',        'wolf'),        'similar'),
        (('computer',   'algorithm'),   'similar'),
        (('scientist',  'researcher'),  'similar'),
        (('athlete',    'champion'),    'similar'),
        (('king',       'computer'),    'dissimilar'),
        (('dog',        'algorithm'),   'dissimilar'),
        (('athlete',    'laboratory'),  'dissimilar'),
    ]
    print(f"  {'Word A':<14}  {'Word B':<14}  {'Cosine':>8}  Expected")
    print(f"  {'─'*52}")
    for (w1, w2), exp in pairs:
        s = model.cosine(w1, w2, vocab)
        sim_str = f"{s:.4f}" if s is not None else "N/A"
        print(f"  {w1:<14}  {w2:<14}  {sim_str:>8}  {exp}")

    sep("NEAREST NEIGHBOURS")
    for word in ['king','queen','man','woman',
                 'scientist','computer','wolf','athlete']:
        if word not in vocab.word2idx:
            continue
        nbrs = model.most_similar(word, vocab, top_k=5)
        ns   = "  ".join(f"{w}({s:.3f})" for w, s in nbrs)
        print(f"  {word:<14} -> {ns}")

    sep("ANALOGY  (pos1 - neg1 + pos2 ~ ?)")
    tests = [
        ('king',   'man',    'woman',      'queen?'),
        ('queen',  'woman',  'man',        'king?'),
        ('palace', 'king',   'laboratory', 'scientist?'),
        ('athlete','sport',  'scientist',  'research?'),
    ]
    for p1, n1, p2, label in tests:
        res = model.analogy(p1, n1, p2, vocab, top_k=5)
        if res:
            top = ", ".join(f"{w}({s:.3f})" for w, s in res[:3])
            print(f"  {p1} - {n1} + {p2}  [{label}]")
            print(f"    -> {top}")

def evaluate_fasttext(ft, vocab, model):
    sep("FASTTEXT  --  OOV SUBWORD EVALUATION")
    oov_tests = [
        ('kingly',         'king'),
        ('queendom',       'queen'),
        ('computerize',    'computer'),
        ('athleticism',    'athlete'),
        ('wolflike',       'wolf'),
        ('scientifically', 'scientist'),
        ('algorithmic',    'algorithm'),
        ('programming',    'programmer'),
    ]
    print(f"  {'OOV Word':<22}  {'Base':<14}  {'FT Sim':>8}  N-gram sample")
    print(f"  {'─'*70}")
    for oov, base in oov_tests:
        sim = ft.cosine(ft.embed(oov), ft.embed(base))
        ngs = sorted(ft.get_ngrams(oov))[:3]
        print(f"  {oov:<22}  {base:<14}  {sim:>8.4f}  {', '.join(ngs)}")
    print(f"\n  FastText <-> Word2Vec consistency:")
    print(f"  {'Word':<16}  {'Sim':>10}")
    print(f"  {'─'*30}")
    for w in ['king','queen','man','woman','computer','athlete','wolf','scientist']:
        if w not in vocab.word2idx:
            continue
        v_ft  = ft.embed(w)
        v_w2v = model.W_in[vocab.word2idx[w]]
        sim   = ft.cosine(v_ft, v_w2v)
        print(f"  {w:<16}  {sim:>10.4f}")


# =============================================================================
# 12. MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 56)
    print("  TASK 4: Word2Vec Skip-Gram + FastText")
    print("  From Raw Text -> Continuous Vector Spaces")
    print("=" * 56)

    print("\n[1/5] Tokenising corpus ...")
    sentences = tokenize(CORPUS_RAW)
    content   = [w for s in sentences for w in s if w not in STOP_WORDS]
    print(f"   Sentences      : {len(sentences)}")
    print(f"   Content tokens : {len(content)}")

    print("\n[2/5] Building vocabulary (min_count=2, stop-words removed) ...")
    vocab = Vocabulary(min_count=2).build(sentences)

    print("\n[3/5] Generating Skip-Gram pairs (window=4, subsampling) ...")
    pairs = generate_skipgram_pairs(sentences, vocab, window=4)
    print(f"   Total (centre, context) pairs: {len(pairs):,}")

    print("\n[4/5] Training Word2Vec -- Skip-Gram + Negative Sampling ...")
    print(f"   Embedding dim  : 100")
    print(f"   Negatives/pair : 10")
    print(f"   Epochs         : 30")
    model = Word2Vec(vocab_size=len(vocab), embed_dim=100,
                     n_negatives=10, lr=0.025)
    model.train(pairs, vocab, epochs=30)

    evaluate(model, vocab)

    print("\n[5/5] Building FastText subword model ...")
    ft = FastTextDemo(n_min=3, n_max=6, embed_dim=100)
    ft.build_from_word2vec(model, vocab)
    evaluate_fasttext(ft, vocab, model)

    print("\nGenerating visualisations ...")
    plot_all(model, vocab, ft, model.loss_history)

    print("\n  Task 4 complete.\n")
