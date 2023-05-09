import view.web as web
import model.decision_tree as dt

if __name__ == '__main__':
    DT, X_train = dt.decision_tree()
    web.check_page(DT, X_train)
