from sklearn.metrics import roc_auc_score
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed


def custom_roc_auc_score(
    df: pd.DataFrame,
    n_jobs: int = -1
) -> float:
    '''
    обязательные поля: user_id, rank {-1, 0, 1}, predict
    '''
    def user_roc_auc_score(
        dataframe: pd.DataFrame
    ) -> float:
        metric_value = 0
        pairs_count = 0
        
        ranks = dataframe['rank'].to_numpy()
        scores = dataframe['predict'].to_numpy()
        
        like_scores = scores[ranks == 1]
        
        ignore_scores = scores[ranks == 0]
        
        dislike_scores = scores[ranks == -1]
        
        for neg_scores, pos_scores in zip(
            [dislike_scores, ignore_scores],
            [np.concatenate((ignore_scores, like_scores)), like_scores]
        ):
            if neg_scores.size and pos_scores.size:
                ranks = np.ones(neg_scores.size + pos_scores.size)
                ranks[: neg_scores.size] = 0
                
                scores = np.concatenate(
                    (neg_scores, pos_scores)
                )
                pairs_count_ = neg_scores.size * pos_scores.size
                metric_value += roc_auc_score(ranks, scores) * pairs_count_
                pairs_count += pairs_count_
        if pairs_count:
            return metric_value / pairs_count
        return 0

    assert tqdm.pandas() is None
    
    groups = df.groupby('user_id')

    user_scores = Parallel(n_jobs=n_jobs)(
        delayed(user_roc_auc_score)(group) for _, group in tqdm(groups, total=len(groups))
    )

    return np.mean(user_scores)
