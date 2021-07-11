import sys

from torch import cuda
from torch.optim import Adam, Adagrad, SGD

from torchkge import LinkPredictionEvaluator
from torchkge.models import TransEModel
from torchkge.sampling import BernoulliNegativeSampler, UniformNegativeSampler, PositionalNegativeSampler, MFSampling, BagSampler
from torchkge.utils import MarginLoss, DataLoader, load_wn18rr, load_fb15k237

from tqdm.autonotebook import tqdm


def main():
    # Load dataset
    kg_train, _, kg_test = load_wn18rr()
    # kg_train, _, kg_test = load_fb15k237()

    # Define some hyper-parameters for training
    emb_dim = 50
    emb_rel = 50
    lr = 0.0006
    n_epochs = 1000
    b_size = 1000
    margin = 5
    cache_dim = 100
    weight_decay = 1e-5
    print("++++++++++ TransE : WN18RR ++++++++++++++")
    print("emb_dim: " + str(emb_dim))
    print("lr: " + str(lr))
    print("n_epochs: " + str(n_epochs))
    print("b_size: " + str(b_size))
    print("margin: " + str(margin))
    print("cache_dim: " + str(cache_dim))
    print("weight_decay: " + str(weight_decay))
    print("++++++++++++++++++++++++++++++++++++++++")

    # Define the model and criterion
    model = TransEModel(emb_dim, kg_train.n_ent, kg_train.n_rel, dissimilarity_type='L1')
    criterion = MarginLoss(margin)

    # Move everything to CUDA if available
    use_cuda = None
    if cuda.is_available():
        cuda.empty_cache()
        model.cuda()
        criterion.cuda()
        use_cuda = 'all'

    # Define the torch optimizer to be used
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # sampler = MFSampling(kg=kg_train, kg_test=kg_test, model=model, cache_dim=cache_dim, n_itter=100, n_factors=250, n_neg=25)
    sampler = BagSampler(kg=kg_train, kg_test=kg_test, n_neg=10)
    # sampler = BernoulliNegativeSampler(kg=kg_train, kg_test=kg_test, n_neg=10)
    dataloader = DataLoader(kg_train, batch_size=b_size, use_cuda=use_cuda)
    evaluator = LinkPredictionEvaluator(model, kg_test)
    iterator = tqdm(range(n_epochs), unit='epoch')
    for epoch in iterator:
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            h, t, r = batch[0], batch[1], batch[2]
            n_h, n_t = sampler.corrupt_batch(h, t, r)

            optimizer.zero_grad()

            # forward + backward + optimize
            pos, neg = model(h, t, n_h, n_t, r)
            loss = criterion(pos, neg)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        if epoch % 25 == 0 and epoch!= 0:
            model.normalize_parameters()
            evaluator.evaluate(100)
            evaluator.print_results(k=1)
            evaluator.print_results(k=3)
            evaluator.print_results(k=10)
            # sampler.update_cache()

        iterator.set_description(
            'Epoch {} | mean loss: {:.5f}'.format(epoch + 1, running_loss / len(dataloader)))

    model.normalize_parameters()
    evaluator.evaluate(100)
    evaluator.print_results(k=1)
    evaluator.print_results(k=3)
    evaluator.print_results(k=10)


if __name__ == "__main__":
    main()
