import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from utils import load_data, EarlyStopping

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    accuracy2 = accuracy_score(labels, prediction)
    micro_precision = precision_score(labels, prediction, average='micro')
    macro_precision = precision_score(labels, prediction, average='macro')
    micro_recall = recall_score(labels, prediction, average='micro')
    macro_recall = recall_score(labels, prediction, average='macro')
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, accuracy2, micro_precision, macro_precision, micro_recall, macro_recall, micro_f1, macro_f1

def evaluate(model, g, pFeatures, queryFeatures, kwFeatures, attFeatures, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g, pFeatures, queryFeatures, kwFeatures, attFeatures)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, accuracy2, micro_precision, macro_precision,\
    micro_recall, macro_recall, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    return loss, accuracy, accuracy2, micro_precision, macro_precision,\
           micro_recall, macro_recall, micro_f1, macro_f1

def main(args):

    g, pFeatures, queryFeatures, kwFeatures, attFeatures, \
    labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
    val_mask, test_mask = load_data(args['dataset'])

    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()

    pFeatures = pFeatures.to(args['device'])
    queryFeatures = queryFeatures.to(args['device'])
    kwFeatures = kwFeatures.to(args['device'])
    attFeatures = attFeatures.to(args['device'])
    labels = labels.to(args['device'])
    train_mask = train_mask.to(args['device'])
    val_mask = val_mask.to(args['device'])
    test_mask = test_mask.to(args['device'])

    from SAN_model import SAN
    model = SAN(meta_paths=[['pq', 'qp'], ['pk', 'kp'], ['pa', 'ap']],
                in_size=pFeatures.shape[1],
                supportFeature_size= {'queryIn_size': queryFeatures.shape[1],
                                      'kwIn_size': kwFeatures.shape[1],
                                      'attIn_size': attFeatures.shape[1]},
                hidden_size=args['hidden_units'],
                out_size=num_classes,
                num_heads=args['num_heads'],
                dropout=args['dropout']).to(args['device'])
    g = g.to(args['device'])

    stopper = EarlyStopping(patience=args['patience'])
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                                 weight_decay=args['weight_decay'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args['num_epochs'], max_lr=args['max_lr'])

    train_step = 0
    for epoch in range(args['num_epochs']):
        model.train()
        logits = model(g, pFeatures, queryFeatures, kwFeatures, attFeatures)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_step += 1
        scheduler.step(train_step)

        train_acc, train_acc2, train_micro_pre, train_macro_pre,\
        train_micro_rec, train_macro_rec, train_micro_f1, train_macro_f1 = score(logits[train_mask], labels[train_mask])
        val_loss, val_acc, val_acc2, val_micro_pre, val_macro_pre,\
        val_micro_rec, val_macro_rec, val_micro_f1, val_macro_f1 = evaluate(model, g, pFeatures, queryFeatures,
                                            kwFeatures, attFeatures, labels, val_mask, loss_fcn)
        early_stop = stopper.step(val_loss.data.item(), val_acc, model)

        print('Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
              'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
            epoch + 1, loss.item(), train_micro_f1, train_macro_f1, val_loss.item(), val_micro_f1, val_macro_f1))

        if early_stop:
            break

    stopper.load_checkpoint(model)
    test_loss, test_acc, test_acc2, test_micro_pre, test_macro_pre, \
    test_micro_rec, test_macro_rec, test_micro_f1, test_macro_f1 = evaluate(model, g, pFeatures, queryFeatures,
                                            kwFeatures, attFeatures, labels, test_mask, loss_fcn)

    print('Test loss {:.4f} | Test Acc {:.4f} | Test Acc2 {:.4f} | Test Micro Pre {:.4f}| Test Macro Pre {:.4f}'.format(
        test_loss, test_acc, test_acc2, test_micro_pre, test_macro_pre))

    print('Test Micro Rec {:.4f}| Test Macro Rec {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
        test_micro_rec, test_macro_rec, test_micro_f1, test_macro_f1))

if __name__ == '__main__':
    import argparse

    from utils import setup

    parser = argparse.ArgumentParser('SAN')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('-ld', '--log-dir', type=str, default='results',
                        help='Dir for saving training results')
    parser.add_argument('--AliCoCo', action='store_true', default=True,
                        help='Use alicoco dataset')
    parser.add_argument('--max_lr', type=float, default=0.005)
    args = parser.parse_args().__dict__

    args = setup(args)

    main(args)
