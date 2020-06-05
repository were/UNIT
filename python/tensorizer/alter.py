from tvm import relay

class AlterOpLayout(object):
    """ Temporarily changes the attr of an op. """
    def __init__(self, op_name, attr_key, attr_value):
        """ Saves the required info for RAII pattern usage.

        Parameters
        ----------
        op_name : str
            The op name.

        attr_key : str
            The attribute name.

        attr_value : object
            The attribute value.

        Examples
        --------
        .. code-block:: python

        # Temporarily update FTVMAlterOpLayout to a user-defined packed function.
        # After the test is finished, the attr value will be set back to the original value.

        with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
            my_mod = relay.transform.AlterOpLayout()(my_mod)

        """
        self.op = relay.op.get(op_name)
        self.attr_key = attr_key
        self.attr_value = attr_value

    def __enter__(self):
        self.older_attr = self.op.get_attr(self.attr_key)
        self.op.reset_attr(self.attr_key)
        self.op.set_attr(self.attr_key, self.attr_value)
        return self

    def __exit__(self, ptype, value, trace):
        self.op.reset_attr(self.attr_key)
        if self.older_attr:
            self.op.set_attr(self.attr_key, self.older_attr)
