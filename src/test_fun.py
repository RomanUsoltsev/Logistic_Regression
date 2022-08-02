import fun


def test_dataset():
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = fun.load_dataset()
    index = 30
    fun.plt.imshow(train_set_x_orig[index])
    print("y = " + str(train_set_y[:, index]) + ", it is class '" +
          classes[fun.np.squeeze(train_set_y[:, index])].decode("utf-8") + "' images.")


def learn_net():
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = fun.load_dataset()

    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    train_set_x = train_set_x_flatten / 255.
    test_set_x = test_set_x_flatten / 255.

    d = fun.model(train_set_x, train_set_y, test_set_x, test_set_y,
                  num_iterations=2000, learning_rate=0.005, print_cost=True)

    costs = fun.np.squeeze(d['costs'])
    fun.plt.plot(costs)
    fun.plt.ylabel('Error')
    fun.plt.xlabel('Number of learn iteration')
    fun.plt.title("Speed learn =" + str(d["learning_rate"]))
    fun.plt.show()


def graph_learn_net():
    learning_rates = [0.01, 0.001, 0.0001]
    models = {}

    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = fun.load_dataset()

    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    train_set_x = train_set_x_flatten / 255.
    test_set_x = test_set_x_flatten / 255.

    for i in learning_rates:
        print("learning rate is: " + str(i))
        models[str(i)] = fun.model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500,
                                   learning_rate=i,
                                   print_cost=False)
        print('\n' + "-------------------------------------------------------" + '\n')

    for i in learning_rates:
        fun.plt.plot(fun.np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

    fun.plt.ylabel('Error')
    fun.plt.xlabel('Number of learn iteration')

    legend = fun.plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    fun.plt.show()


from skimage import transform
from skimage.io import imread


def user_check_image(path):
    image = fun.np.array(imread(path))
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = fun.load_dataset()

    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    train_set_x = train_set_x_flatten / 255.
    test_set_x = test_set_x_flatten / 255.

    d = fun.model(train_set_x, train_set_y, test_set_x, test_set_y,
                  num_iterations=2000, learning_rate=0.005, print_cost=True)
    num_px = train_set_x_orig.shape[1]
    num_py = train_set_x_orig.shape[2]

    image = image / 255.
    image = transform.resize(image, [num_py, num_px, 3])
    my_image = image.reshape((1, num_py * num_px * 3)).T

    my_predicted_image = fun.predict(d["w"], d["b"], my_image)

    fun.plt.imshow(image)
    print("y = " + str(fun.np.squeeze(my_predicted_image)) + ", neural network prediction: \"" + classes[
        int(fun.np.squeeze(my_predicted_image)),].decode("utf-8") + "\" on the image.")
