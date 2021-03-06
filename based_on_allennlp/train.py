from allennlp.training.trainer import GradientDescentTrainer
from allennlp.training.optimizers import AdamOptimizer
from allennlp_models.generation import Seq2SeqPredictor
import itertools


def train(args, model, dataset_reader, train_loader, device=0,
          val_loader=None, test_data=None, num_epochs=10, patience=None, serialization_dir=None):
    optimizer = AdamOptimizer(model.named_parameters(), lr=args.lr, weight_decay=args.l2)
    trainer = GradientDescentTrainer(model=model,
                                     optimizer=optimizer,
                                     data_loader=train_loader,
                                     validation_data_loader=val_loader,
                                     cuda_device=device,
                                     num_epochs=num_epochs,
                                     serialization_dir=serialization_dir,
                                     patience=patience,
                                     grad_clipping=args.clip,
                                     )
    trainer.train()

    if test_data is not None:
        predictor = Seq2SeqPredictor(model, dataset_reader)
        for instance in itertools.islice(test_data, 10):
            print('SOURCE:', instance.fields['source_tokens'].tokens)
            print('GOLD:', instance.fields['target_tokens'].tokens)
            print('PRED:', predictor.predict_instance(instance)['predicted_tokens'])
            print('-' * 50)

