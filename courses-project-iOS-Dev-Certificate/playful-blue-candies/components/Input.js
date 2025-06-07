import { StyleSheet, Text, TextInput, View } from 'react-native';
import { COLORS } from '../constants/theme';
import useFontSize from '../hooks/useFontSize';

export const Input = ({ style, error, ...otherProps }) => {
  const fontStyle = useFontSize(16);

  return (
    <View style={styles.inputContainer}>
      <TextInput
        clearButtonMode="always"
        placeholderTextColor={COLORS.lightGray}
        style={[
          styles.input,
          {
            height: fontStyle.lineHeight + 16,
            fontSize: fontStyle.fontSize,
          },
          style,
        ]}
        {...otherProps}
      />
      {!!error && <Text style={[styles.errorText, fontStyle]}>{error}</Text>}
    </View>
  );
};

const styles = StyleSheet.create({
  inputContainer: {
    gap: 10,
  },
  input: {
    borderRadius: 8,
    padding: 8,
    borderWidth: 1,
    borderColor: COLORS.darkGray,
  },
  errorText: {
    color: COLORS.red,
  },
});
