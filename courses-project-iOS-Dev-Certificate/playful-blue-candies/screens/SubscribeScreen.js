import { useState } from 'react';
import {
  Alert,
  Image,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StyleSheet,
  View,
} from 'react-native';
import { Button } from '../components/Button';
import { Input } from '../components/Input';
import { BaseText } from '../components/Typography';
import { BASE_SCREEN_STYLE } from '../constants/theme';
import { debounce, validateEmail } from '../utils/index';

const SubscribeScreen = () => {
  const [email, onChangeEmail] = useState('');
  const [invalidMailMsg, setinvalidMailMsg] = useState('');

  const validateEmailHandler = debounce((mail) =>
    setinvalidMailMsg(validateEmail(mail) ? '' : 'Invalid email')
  );

  const onChangeEmailHandler = (email) => {
    onChangeEmail(email);
    validateEmailHandler(email);
  };

  const clearInput = () => {
    onChangeEmail('');
    setinvalidMailMsg('');
  };

  const subscriptionHandler = () => {
    Alert.alert(`Thanks for subscribing ${email}, stay tuned!`);
    clearInput();
  };

  return (
    <View style={styles.screenContainer}>
      <Image
        source={require('../assets/little-lemon-logo-grey.png')}
        style={styles.logo}
      />
      <BaseText>
        Subscribe to our newsLetter for our latest delicious recipies!
      </BaseText>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}>
        <ScrollView
          contentContainerStyle={styles.formContainer}
          keyboardDismissMode="on-drag">
          <Input
            placeholder="Enter your email"
            value={email}
            onChangeText={onChangeEmailHandler}
            error={invalidMailMsg}
            keyboardType="email-address"
            inputMode="email"
            textContentType="emailAddress"
          />
          <Button
            disabled={email?.length < 6 || !!invalidMailMsg}
            onPress={subscriptionHandler}>
            Subscribe
          </Button>
        </ScrollView>
      </KeyboardAvoidingView>
    </View>
  );
};

export default SubscribeScreen;

const styles = StyleSheet.create({
  screenContainer: { ...BASE_SCREEN_STYLE, gap: 24 },
  logo: {
    resizeMode: 'contain',
    width: 'auto',
    height: 100,
  },
  formContainer: {
    gap: 24,
  },
});
