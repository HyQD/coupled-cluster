class AlphaMixer:
    """Basic mixer class

    Parameters
    ----------
    theta : float
        Mixing parameter. Must be in [0, 1]
    np : module
        Matrix library to be used, e.g., numpy, cupy, etc.
    """

    def __init__(self, theta=0.1, np=None):
        assert 0 <= theta <= 1, "Mixing parameter theta must be in [0, 1]"

        self.theta = theta

    def compute_new_vector(self, trial_vector, direction_vector, error_vector):
        """Compute new trial vector for mixing with full right hand side.

        See T. Helgaker's book "Molecular Electron-Structure Theory" equations
        (13.4.3), (13.4.6) and (13.4.10).

        Parameters
        ----------
        trial_vector : np.array
            Inital vector for mixing.
        direction_vector : np.array
            Vector to be addet to trial_vector.
        error_vector : np.array
            Not used in alpha mixer, for DIIS mixer.

        Returns
        -------
        np.array
            New mixed vector.
        """
        new_trial = trial_vector + direction_vector

        return (1 - self.theta) * new_trial + self.theta * trial_vector

    def clear_vectors(self):
        pass


class DIIS(AlphaMixer):
    """Direct Inversion in Iterative Subspace (DIIS)

    General vector mixing class to accelerate quasi-Newton
    using direct inversion of iterative space.

    Code inherited from Simen Kvaal.

    Parameters
    ----------
    np : module
        Matrix library to be used, e.g., numpy, cupy, etc.
    num_vecs : int
        Number of vectors to keep in memory. Default is ``10``.
    """

    def __init__(self, np, num_vecs=10):
        self.np = np
        self.num_vecs = num_vecs
        self.stored = 0

        self.trial_vectors = [0] * self.num_vecs
        self.direction_vectors = [0] * self.num_vecs
        self.error_vectors = [0] * self.num_vecs

    def compute_new_vector(self, trial_vector, direction_vector, error_vector):
        """DIIS mixing scheme

        Parameters
        ----------
        trial_vector : np.array
            Inital vector for mixing.
        direction_vector : np.array
            Vector to be addet to trial_vector.
        error_vector : np.array

        Returns
        -------
        np.array
            New mixed vector
        """

        np = self.np

        new_pos = self.stored % self.num_vecs
        self.stored += 1

        self.trial_vectors[new_pos] = trial_vector.ravel()
        self.direction_vectors[new_pos] = direction_vector.ravel()
        self.error_vectors[new_pos] = error_vector.ravel()

        b_dim = self.stored if self.stored < self.num_vecs else self.num_vecs

        b_vec = np.zeros(b_dim + 1, dtype=trial_vector.dtype)
        b_mat = np.zeros((b_dim + 1, b_dim + 1), dtype=trial_vector.dtype)

        for i in range(b_dim):
            for j in range(i + 1):
                b_mat[i, j] = np.dot(
                    self.error_vectors[i], self.error_vectors[j]
                )

                if i != j:
                    b_mat[j, i] = b_mat[i, j]

            b_mat[i, b_dim] = -1.0
            b_mat[b_dim, i] = -1.0

        b_vec[b_dim] = -1.0
        pre_condition = np.zeros_like(b_vec)

        if np.any(np.diag(b_mat)[:-1] <= 0):
            pre_condition[:-1] = 1
        else:
            pre_condition[:-1] += np.power(np.diag(b_mat)[:-1], -0.5)

        pre_condition[b_dim] = 1

        for i in range(b_dim + 1):
            for j in range(b_dim + 1):
                b_mat[i, j] *= pre_condition[i] * pre_condition[j]

        weights = -np.linalg.pinv(b_mat)[b_dim]
        weights[:-1] *= pre_condition[:-1]

        new_trial_vector = np.zeros_like(self.trial_vectors[new_pos])

        for i in range(b_dim):
            new_trial_vector += weights[i] * (
                self.trial_vectors[i] + self.direction_vectors[i]
            )

        return new_trial_vector.reshape(trial_vector.shape)

    def clear_vectors(self):
        """
        Delete all stored vectors and start fresh.
        """

        self.trial_vectors = [0] * self.num_vecs
        self.direction_vectors = [0] * self.num_vecs
        self.error_vectors = [0] * self.num_vecs

        self.stored = 0
